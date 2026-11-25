#include "stream-whisper.h"

#include <cstdio>
#include <string>
#include <vector>

#include "whisper.h"

namespace whisper_stream {

whisper_transcriber * whisper_transcriber::create(const whisper_transcriber_config & cfg,
                                                  std::string & err) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = cfg.use_gpu;
    cparams.flash_attn = cfg.flash_attn;

    whisper_context * ctx =
        whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    if (ctx == nullptr) {
        err = "failed to initialize whisper context from '" + cfg.model_path + "'";
        return nullptr;
    }

    return new whisper_transcriber(ctx, cfg);
}

whisper_transcriber::whisper_transcriber(whisper_context * ctx,
                                         const whisper_transcriber_config & cfg)
    : ctx_(ctx), cfg_(cfg) {}

whisper_transcriber::~whisper_transcriber() {
    if (ctx_ != nullptr) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

int32_t whisper_transcriber::eot_id() const {
    return ctx_ != nullptr ? whisper_token_eot(ctx_) : 0;
}

int32_t whisper_transcriber::text_ctx() const {
    return ctx_ != nullptr ? whisper_n_text_ctx(ctx_) : 0;
}

bool whisper_transcriber::transcribe(const float * samples, int n_samples,
                                     int64_t /*window_start_ms*/,
                                     const std::vector<int32_t> & prompt_tokens,
                                     std::vector<token_time> & out,
                                     std::string & err) {
    if (ctx_ == nullptr) {
        err = "transcriber has no whisper context";
        return false;
    }

    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.token_timestamps = true;
    wparams.single_segment   = false;
    wparams.no_context       = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.offset_ms        = 0;

    wparams.n_threads        = cfg_.n_threads;
    wparams.language         = cfg_.language.c_str();
    wparams.translate        = cfg_.translate;
    wparams.tdrz_enable      = cfg_.tinydiarize;
    wparams.temperature_inc  = cfg_.no_fallback ? 0.0f : wparams.temperature_inc;
    wparams.print_timestamps = false;
    wparams.print_special    = false;

    // the session owns prompt management; no_context stays true so the
    // library itself never reuses previous decoder state
    prompt_storage_.assign(prompt_tokens.begin(), prompt_tokens.end());
    wparams.prompt_tokens   = prompt_storage_.empty() ? nullptr
                                                       : prompt_storage_.data();
    wparams.prompt_n_tokens = static_cast<int>(prompt_storage_.size());

    if (cfg_.vad) {
        wparams.vad            = true;
        wparams.vad_model_path = cfg_.vad_model_path.c_str();
        wparams.vad_params.min_silence_duration_ms =
            static_cast<int>(cfg_.vad_min_silence_ms);
    }

    const int rc = whisper_full(ctx_, wparams, samples, n_samples);
    if (rc != 0) {
        err = "whisper_full() failed";
        return false;
    }

    out.clear();

    const int n_segments = whisper_full_n_segments(ctx_);
    for (int s = 0; s < n_segments; ++s) {
        const bool turn = whisper_full_get_segment_speaker_turn_next(ctx_, s);
        const int  n_tok = whisper_full_n_tokens(ctx_, s);

        for (int i = 0; i < n_tok; ++i) {
            token_time t;
            t.id = whisper_full_get_token_id(ctx_, s, i);
            // always the raw (pre-VAD) timeline; the session adds the
            // window offset itself
            t.t0_cs = whisper_full_get_token_t0(ctx_, s, i);
            t.t1_cs = whisper_full_get_token_t1(ctx_, s, i);

            const char * text = whisper_full_get_token_text(ctx_, s, i);
            t.text = text != nullptr ? std::string(text) : std::string();

            t.speaker_turn_next = turn && i == 0;
            out.push_back(t);
        }
    }

    return true;
}

} // namespace whisper_stream
