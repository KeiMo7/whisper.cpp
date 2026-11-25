#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "common-whisper.h"
#include "stream-session.h"
#include "stream-whisper.h"
#include "whisper.h"

using namespace whisper_stream;

namespace {

int g_checks = 0;

#define REQUIRE(cond)                                                       \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            fprintf(stderr, "REQUIRE failed at %s:%d: %s\n",                \
                    __FILE__, __LINE__, #cond);                             \
            return 1;                                                       \
        }                                                                   \
    } while (0)

} // namespace

int main() {
    ggml_backend_load_all();

    const std::string model_path  = WHISPER_MODEL_PATH;
    const std::string sample_path = SAMPLE_PATH;

    std::vector<float>              pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    REQUIRE(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    REQUIRE(!pcmf32.empty());

    const int64_t duration_ms = samples_to_ms(static_cast<int64_t>(pcmf32.size()));
    REQUIRE(duration_ms > 0);

    whisper_transcriber_config tcfg;
    tcfg.model_path = model_path;
    tcfg.n_threads  = 4;

    std::string             err;
    whisper_transcriber *  transcriber = whisper_transcriber::create(tcfg, err);
    REQUIRE(transcriber != nullptr);

    session_config cfg;
    cfg.step_ms       = 3000;
    cfg.window_ms     = 10000;
    cfg.keep_ms       = 200;
    cfg.emit_partials = false;

    int             n_finals = 0;
    int             n_bad    = 0;
    int64_t         prev_t1  = 0;
    bool            failed   = false;
    std::string     collected;

    auto sink = [&](const event & e) {
        if (e.kind == event_kind::transcribe_failed) {
            failed = true;
        }
        if (e.kind != event_kind::final_segment) {
            return;
        }
        if (!(e.seg.t0_ms < e.seg.t1_ms) ||
            !(e.seg.t0_ms >= prev_t1) ||
            !(e.seg.t1_ms <= duration_ms + 50)) {
            ++n_bad;
        }
        prev_t1 = e.seg.t1_ms;
        ++n_finals;
        collected += e.seg.text;
    };

    stream_session session(cfg, *transcriber, sink);

    const int chunk = static_cast<int>(
        100 * WHISPER_STREAM_SAMPLE_RATE / 1000);

    size_t  pos          = 0;
    int64_t fed_ms       = 0;
    int64_t last_step_ms = 0;

    while (pos < pcmf32.size()) {
        const size_t n = std::min(static_cast<size_t>(chunk),
                                  pcmf32.size() - pos);
        REQUIRE(session.push_audio(pcmf32.data() + pos,
                                   static_cast<int>(n)) != push_status::closed);
        pos    += n;
        fed_ms += samples_to_ms(static_cast<int64_t>(n));

        if (fed_ms - last_step_ms >= cfg.step_ms) {
            session.step();
            last_step_ms = fed_ms;
        }
    }

    session.finish();

    REQUIRE(!failed);
    REQUIRE(n_bad == 0);
    REQUIRE(session.get_stats().windows > 0);

    // finish() is idempotent and produces no further events
    const int events_before = n_finals;
    REQUIRE(session.finish() == 0);
    REQUIRE(n_finals == events_before);
    REQUIRE(session.get_stats().finish_calls == 2);

    // writes after finish are rejected
    float one = 0.0f;
    REQUIRE(session.push_audio(&one, 1) == push_status::closed);

    if (n_finals == 0) {
        // The shipped for-tests model is a weightless stub ("assuming empty
        // model for testing") and emits no tokens. In that case only the
        // pipeline mechanics (scheduling, finish, no failures, no crashes)
        // can be exercised; the content assertions below run against a real
        // model, e.g. models/ggml-tiny.en.bin.
        printf("note: model produced no tokens (stub test model); "
               "skipping content assertions\n");
        printf("duration_ms=%lld windows=%lld lag_ms=%lld\n",
               static_cast<long long>(duration_ms),
               static_cast<long long>(session.get_stats().windows),
               static_cast<long long>(session.get_stats().lag_ms));
        delete transcriber;
        printf("ok - stream-session-whisper (pipeline only, %d checks)\n",
               g_checks);
        return 0;
    }

    // Content assertions, with a real model.
    REQUIRE(!collected.empty());
    REQUIRE(session.get_stats().committed_ms >= 8000);
    REQUIRE(session.get_stats().lag_ms <= 2000);

    printf("duration_ms=%lld finals=%d committed_ms=%lld lag_ms=%lld text=\"%s\"\n",
           static_cast<long long>(duration_ms), n_finals,
           static_cast<long long>(session.get_stats().committed_ms),
           static_cast<long long>(session.get_stats().lag_ms),
           collected.c_str());

    delete transcriber;
    printf("ok - stream-session-whisper (%d checks)\n", g_checks);
    return 0;
}
