// Offline, timestamp-aligned transcription built on the SDL-free
// stream-session library. Emits one "[HH:MM:SS.mmm --> HH:MM:SS.mmm]  text"
// line per committed segment, suitable for pipes and parsers.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "common-whisper.h"
#include "stream-session.h"
#include "stream-whisper.h"
#include "whisper.h"

using namespace whisper_stream;

namespace {

struct align_params {
    int32_t     n_threads = 4;
    int64_t     step_ms   = 3000;
    int64_t     length_ms = 10000;
    int64_t     keep_ms   = 200;
    int64_t     chunk_ms  = 100;
    bool        partials  = false;
    std::string model     = "models/for-tests-ggml-tiny.en.bin";
    std::string file;
    std::string output;
};

void print_usage(const char * prog) {
    fprintf(stderr, "\nusage: %s [options]\n\n", prog);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model FNAME    model path\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     input audio file\n");
    fprintf(stderr, "  -t N,     --threads N      number of threads\n");
    fprintf(stderr, "            --step N         scheduling step in ms [3000]\n");
    fprintf(stderr, "            --length N       window length in ms [10000]\n");
    fprintf(stderr, "            --keep N         overlap kept after commit in ms [200]\n");
    fprintf(stderr, "            --chunk N        feed chunk size in ms [100]\n");
    fprintf(stderr, "            --partials       also emit partial segments\n");
    fprintf(stderr, "  -o FNAME, --output FNAME   write output to this file\n");
    fprintf(stderr, "\n");
}

bool parse_args(int argc, char ** argv, align_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { return false; }
            p.model = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) { return false; }
            p.file = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) { return false; }
            p.n_threads = std::stoi(argv[i]);
        } else if (arg == "--step") {
            if (++i >= argc) { return false; }
            p.step_ms = std::stoll(argv[i]);
        } else if (arg == "--length") {
            if (++i >= argc) { return false; }
            p.length_ms = std::stoll(argv[i]);
        } else if (arg == "--keep") {
            if (++i >= argc) { return false; }
            p.keep_ms = std::stoll(argv[i]);
        } else if (arg == "--chunk") {
            if (++i >= argc) { return false; }
            p.chunk_ms = std::stoll(argv[i]);
        } else if (arg == "--partials") {
            p.partials = true;
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { return false; }
            p.output = argv[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (p.file.empty()) {
        fprintf(stderr, "error: -f/--file is required\n");
        return false;
    }
    if (p.step_ms <= 0 || p.length_ms <= 0 || p.keep_ms < 0 ||
        p.chunk_ms <= 0) {
        fprintf(stderr, "error: step, length and chunk must be positive, keep non-negative\n");
        return false;
    }
    p.keep_ms   = std::min(p.keep_ms, p.step_ms);
    p.length_ms = std::max(p.length_ms, p.step_ms);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    align_params params;
    if (!parse_args(argc, argv, params)) {
        if (argc >= 2 &&
            (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            return 0;
        }
        print_usage(argv[0]);
        return 1;
    }

    std::ofstream fout;
    if (!params.output.empty()) {
        fout.open(params.output);
        if (!fout.is_open()) {
            fprintf(stderr, "error: failed to open output file '%s'\n",
                    params.output.c_str());
            return 1;
        }
    }

    std::vector<float>              pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(params.file, pcmf32, pcmf32s, false) || pcmf32.empty()) {
        fprintf(stderr, "error: failed to read audio from '%s'\n",
                params.file.c_str());
        return 3;
    }

    whisper_transcriber_config tcfg;
    tcfg.model_path = params.model;
    tcfg.n_threads  = params.n_threads;

    std::string            err;
    whisper_transcriber * transcriber = whisper_transcriber::create(tcfg, err);
    if (transcriber == nullptr) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 2;
    }

    session_config scfg;
    scfg.step_ms       = params.step_ms;
    scfg.window_ms     = params.length_ms;
    scfg.keep_ms       = params.keep_ms;
    scfg.emit_partials = params.partials;

    bool failed = false;

    auto sink = [&](const event & e) {
        if (e.kind == event_kind::transcribe_failed) {
            failed = true;
            return;
        }
        if (e.kind != event_kind::final_segment &&
            e.kind != event_kind::partial_segment) {
            return;
        }

        // to_timestamp expects centiseconds; t_ms/10 truncates as required
        const std::string line =
            "[" + to_timestamp(e.seg.t0_ms / 10, false) +
            " --> " + to_timestamp(e.seg.t1_ms / 10, false) + "]" +
            (e.seg.final ? "  " : "~ ") + e.seg.text + "\n";

        fputs(line.c_str(), stdout);
        if (fout.is_open()) {
            fout << line;
        }
    };

    stream_session session(scfg, *transcriber, sink);

    const int   chunk_samples = static_cast<int>(
        params.chunk_ms * WHISPER_STREAM_SAMPLE_RATE / 1000);
    size_t        pos           = 0;
    const size_t  total         = pcmf32.size();
    int64_t       fed_ms        = 0;
    int64_t       last_step_ms  = 0;

    while (pos < total) {
        const size_t n = std::min(static_cast<size_t>(chunk_samples), total - pos);
        if (session.push_audio(pcmf32.data() + pos, static_cast<int>(n))
            == push_status::closed) {
            break;
        }
        pos    += n;
        fed_ms += samples_to_ms(static_cast<int64_t>(n));

        if (fed_ms - last_step_ms >= params.step_ms) {
            session.step();
            last_step_ms = fed_ms;
        }
    }

    session.finish();

    if (fout.is_open()) {
        fout.close();
    }
    delete transcriber;

    return failed ? 6 : 0;
}
