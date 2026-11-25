#pragma once

// stream_transcriber implementation backed by whisper_full().

#include <cstdint>
#include <string>
#include <vector>

#include "stream-session.h"

namespace whisper_stream {

struct whisper_transcriber_config {
    int32_t     n_threads          = 4;
    bool        translate          = false;
    bool        no_fallback        = false;
    bool        tinydiarize        = false;
    bool        flash_attn         = true;
    bool        use_gpu            = true;
    bool        vad                = false;
    int64_t     vad_min_silence_ms = 300;
    std::string language           = "en";
    std::string model_path;
    std::string vad_model_path;
};

class whisper_transcriber : public stream_transcriber {
public:
    // Returns nullptr (and fills err) when the model cannot be loaded.
    static whisper_transcriber * create(const whisper_transcriber_config & cfg,
                                        std::string & err);

    ~whisper_transcriber() override;

    bool transcribe(const float * samples, int n_samples,
                    int64_t window_start_ms,
                    const std::vector<int32_t> & prompt_tokens,
                    std::vector<token_time> & out,
                    std::string & err) override;

    int32_t eot_id()  const override;
    int32_t text_ctx() const override;

private:
    explicit whisper_transcriber(struct whisper_context * ctx,
                                 const whisper_transcriber_config & cfg);

    struct whisper_context *         ctx_;
    whisper_transcriber_config       cfg_;
    std::vector<whisper_token>       prompt_storage_;
};

} // namespace whisper_stream
