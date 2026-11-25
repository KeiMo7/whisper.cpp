// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "stream-session.h"
#include "stream-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = true;

    bool align         = false; // session-based, timestamped, parseable output
    bool partials      = false; // emit partial segments in align mode

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn    = false; }
        else if (                  arg == "--align")         { params.align         = true; }
        else if (                  arg == "--partials")      { params.partials      = true; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention during inference\n",        params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn [%-7s] disable flash attention during inference\n",       params.flash_attn ? "false" : "true");
    fprintf(stderr, "            --align         [%-7s] timestamped, parseable line output (no ANSI)\n",   params.align ? "true" : "false");
    fprintf(stderr, "            --partials      [%-7s] also emit partial segments (requires --align)\n",  params.partials ? "true" : "false");
    fprintf(stderr, "\n");
}

// Legacy sliding-window VAD mode (--step 0): unchanged behavior.
static int run_legacy_vad(whisper_params & params, audio_async & audio,
                          struct whisper_context * ctx, std::ofstream & fout) {
    const int n_samples_30s = (int) (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    std::vector<whisper_token> prompt_tokens;

    int  n_iter     = 0;
    bool is_running = true;

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        const auto t_now  = std::chrono::high_resolution_clock::now();
        const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

        if (t_diff < 2000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        audio.get(2000, pcmf32_new);

        if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
            audio.get(params.length_ms, pcmf32);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        t_last = t_now;

        whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

        wparams.print_progress   = false;
        wparams.print_special    = params.print_special;
        wparams.print_realtime   = false;
        wparams.print_timestamps = true;
        wparams.translate        = params.translate;
        wparams.single_segment   = false;
        wparams.max_tokens       = params.max_tokens;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.audio_ctx        = params.audio_ctx;
        wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]
        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

        wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
        wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "failed to process audio\n");
            return 6;
        }

        {
            const int64_t t1 = (t_last - t_start).count()/1000000;
            const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

            printf("\n");
            printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
            printf("\n");

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);

                const int64_t st0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t st1 = whisper_full_get_segment_t1(ctx, i);

                std::string output = "[" + to_timestamp(st0, false) + " --> " + to_timestamp(st1, false) + "]  " + text;

                if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                    output += " [SPEAKER_TURN]";
                }

                output += "\n";

                printf("%s", output.c_str());
                fflush(stdout);

                if (params.fname_out.length() > 0) {
                    fout << output;
                }
            }

            if (params.fname_out.length() > 0) {
                fout << std::endl;
            }

            printf("\n");
            printf("### Transcription %d END\n", n_iter);
        }

        ++n_iter;

        if (!params.no_context) {
            prompt_tokens.clear();

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const int token_count = whisper_full_n_tokens(ctx, i);
                for (int j = 0; j < token_count; ++j) {
                    prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                }
            }
        }

        fflush(stdout);
    }

    return 0;
}

// Session-based path: default rolling single-segment output, or --align.
static int run_session(whisper_params & params, audio_async & audio,
                       whisper_stream::whisper_transcriber & transcriber,
                       std::ofstream & fout) {
    using namespace whisper_stream;

    session_config scfg;
    scfg.step_ms       = params.step_ms;
    scfg.window_ms     = params.length_ms;
    scfg.keep_ms       = params.keep_ms;
    scfg.emit_partials = params.align && params.partials;

    std::mutex out_mutex;
    bool       failed = false;

    auto sink = [&](const event & e) {
        std::lock_guard<std::mutex> lock(out_mutex);

        if (e.kind == event_kind::transcribe_failed) {
            failed = true;
            return;
        }
        if (e.kind != event_kind::final_segment &&
            e.kind != event_kind::partial_segment) {
            return;
        }

        if (params.align) {
            const std::string line =
                "[" + to_timestamp(e.seg.t0_ms / 10, false) +
                " --> " + to_timestamp(e.seg.t1_ms / 10, false) + "]" +
                (e.seg.final ? "  " : "~ ") + e.seg.text + "\n";
            fputs(line.c_str(), stdout);
            if (fout.is_open()) {
                fout << line;
            }
            fflush(stdout);
            return;
        }

        // default path: one rolling segment, no timestamps, ANSI erase line
        if (e.kind == event_kind::final_segment) {
            printf("\33[2K\r%s", e.seg.text.c_str());
            if (fout.is_open()) {
                fout << e.seg.text;
            }
        } else {
            printf("\33[2K\r%s", e.seg.text.c_str());
        }
        fflush(stdout);
    };

    stream_session session(scfg, transcriber, sink);

    const int step_samples =
        (int) (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;

    std::vector<float> chunk;
    bool               is_running = true;

    while (is_running) {
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        audio.get(params.step_ms, chunk);

        if ((int) chunk.size() > 2 * step_samples) {
            fprintf(stderr, "\n\nWARNING: cannot process audio fast enough, dropping audio ...\n\n");
            audio.clear();
            chunk.clear();
            continue;
        }

        if ((int) chunk.size() < step_samples) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        audio.clear();

        session.push_audio(chunk.data(), (int) chunk.size());
        session.step();
    }

    session.finish();
    fflush(stdout);

    if (!params.align) {
        printf("\n");
    }

    return failed ? 6 : 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const bool use_vad = params.step_ms <= 0; // sliding window mode uses VAD

    if (params.align && use_vad) {
        fprintf(stderr, "error: --align cannot be combined with --step 0\n");
        return 1;
    }

    // init audio

    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    int ret = 0;

    if (use_vad) {
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu    = params.use_gpu;
        cparams.flash_attn = params.flash_attn;

        struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to initialize whisper context\n");
            return 2;
        }

        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        fprintf(stderr, "\n");

        printf("[Start speaking]\n");
        fflush(stdout);

        ret = run_legacy_vad(params, audio, ctx, fout);

        audio.pause();
        whisper_print_timings(ctx);
        whisper_free(ctx);
        return ret;
    }

    // session-based path (default rolling output or --align)
    whisper_stream::whisper_transcriber_config tcfg;
    tcfg.model_path       = params.model;
    tcfg.n_threads        = params.n_threads;
    tcfg.translate        = params.translate;
    tcfg.no_fallback      = params.no_fallback;
    tcfg.tinydiarize      = params.tinydiarize;
    tcfg.use_gpu          = params.use_gpu;
    tcfg.flash_attn       = params.flash_attn;
    tcfg.language         = params.language;

    std::string                              tr_err;
    whisper_stream::whisper_transcriber * transcriber =
        whisper_stream::whisper_transcriber::create(tcfg, tr_err);
    if (transcriber == nullptr) {
        fprintf(stderr, "error: %s\n", tr_err.c_str());
        return 2;
    }

    const int n_new_line = std::max(1, params.length_ms / params.step_ms - 1);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: processing (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s ...\n",
            __func__,
            float(params.step_ms)/1000.0f,
            float(params.length_ms)/1000.0f,
            float(params.keep_ms)/1000.0f,
            params.n_threads,
            params.language.c_str(),
            params.translate ? "translate" : "transcribe");
    fprintf(stderr, "%s: n_new_line = %d, align = %d\n", __func__, n_new_line, params.align);
    fprintf(stderr, "\n");

    printf("[Start speaking]\n");
    fflush(stdout);

    ret = run_session(params, audio, *transcriber, fout);

    audio.pause();
    delete transcriber;
    return ret;
}
