#include <atomic>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio-ring.h"
#include "stream-session.h"

using namespace whisper_stream;

namespace {

int g_checks = 0;

#define REQUIRE(cond)                                                       \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            fprintf(stderr, "REQUIRE failed at %s:%d: %s\n",                \
                    __FILE__, __LINE__, #cond);                             \
            throw std::runtime_error("test failure");                       \
        }                                                                   \
    } while (0)

// Scripted transcriber: returns queued hypotheses in order, or failures.
class fake_transcriber : public stream_transcriber {
public:
    std::vector<std::vector<token_time>> scripts;
    std::vector<bool>                    fail_flags;
    bool                                 throw_next = false;
    std::vector<int32_t>                 last_prompt;
    int32_t                              eot = 50257;
    int32_t                              ctx = 448;
    int                                  calls = 0;

    bool transcribe(const float * /*samples*/, int n_samples,
                    int64_t /*window_start_ms*/,
                    const std::vector<int32_t> & prompt_tokens,
                    std::vector<token_time> & out,
                    std::string & err) override {
        last_prompt = prompt_tokens;
        ++calls;

        if (throw_next) {
            throw_next = false;
            throw std::runtime_error("scripted exception");
        }

        const size_t idx = static_cast<size_t>(calls - 1);
        if (idx < fail_flags.size() && fail_flags[idx]) {
            err = "scripted failure";
            return false;
        }
        if (n_samples <= 0) {
            err = "no samples";
            return false;
        }

        out = idx < scripts.size() ? scripts[idx]
                                   : std::vector<token_time>{};
        return true;
    }

    int32_t eot_id()  const override { return eot; }
    int32_t text_ctx() const override { return ctx; }
};

token_time make_tok(int32_t id, int64_t t0_cs, int64_t t1_cs,
                    const std::string & text, bool turn = false) {
    token_time t;
    t.id                = id;
    t.t0_cs             = t0_cs;
    t.t1_cs             = t1_cs;
    t.text              = text;
    t.speaker_turn_next = turn;
    return t;
}

int64_t ms_samples(int64_t ms) {
    return ms * WHISPER_STREAM_SAMPLE_RATE / 1000;
}

void feed_ms(stream_session & s, int64_t ms) {
    const int n = static_cast<int>(ms_samples(ms));
    std::vector<float> buf(n > 0 ? n : 1, 0.0f);
    REQUIRE(s.push_audio(buf.data(), n) == push_status::ok);
}

std::vector<event> run_collecting(const session_config & cfg,
                                  fake_transcriber & tr,
                                  const std::function<void(stream_session &)> & driver) {
    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });
    driver(session);
    return events;
}

// ---------- basic time arithmetic ----------

void test_time_conversion() {
    REQUIRE(samples_to_ms(16000) == 1000);
    REQUIRE(samples_to_ms(15999) == 999);
    REQUIRE(samples_to_ms(0) == 0);
    REQUIRE(samples_to_ms(-16000) == -1000);
    REQUIRE(cs_to_ms(120) == 1200);
    REQUIRE(cs_to_ms(0) == 0);
}

// ---------- alignment: window offset is added to token times ----------

void test_alignment() {
    session_config cfg;
    cfg.tail_guard_ms = 0; // make the tokens committable immediately

    // hypotheses at anchor 0 keep distinct ids (no early commit); once the
    // window is clipped to anchor 3000 the ids become identical so two
    // consecutive hypotheses commit there.
    struct anchor_transcriber : stream_transcriber {
        int calls = 0;
        bool transcribe(const float *, int, int64_t ws,
                        const std::vector<int32_t> &,
                        std::vector<token_time> & out,
                        std::string &) override {
            ++calls;
            token_time t = make_tok(ws == 3000 ? 1 : (100 + calls),
                                    120, 250, " hello");
            out = { t };
            return true;
        }
        int32_t eot_id()  const override { return 50257; }
        int32_t text_ctx() const override { return 448; }
    } tr;

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    feed_ms(session, 5000);
    session.step();                       // anchor 0, hypothesis 1
    feed_ms(session, 5000);
    session.step();                       // anchor 0, hypothesis 2
    feed_ms(session, 3000);               // fed 13000 -> clip anchor to 3000
    session.step();                       // anchor 3000, hypothesis 1
    session.step();                       // same anchor (no new audio) -> commit

    bool found = false;
    for (const event & e : events) {
        if (e.kind == event_kind::final_segment &&
            e.seg.t0_ms == 4200 && e.seg.t1_ms == 5500) {
            found = true;
        }
    }
    REQUIRE(found);
    session.finish();
}

// ---------- stable prefix LCP ----------

void test_stable_prefix() {
    session_config cfg;
    cfg.tail_guard_ms = 0;
    cfg.emit_partials = true;

    fake_transcriber tr;
    tr.scripts.push_back({
        make_tok(1, 10, 20, " a"),
        make_tok(2, 20, 30, " b"),
        make_tok(3, 30, 40, " c"),
    });
    tr.scripts.push_back({
        make_tok(1, 10, 20, " a"),
        make_tok(2, 20, 30, " b"),
        make_tok(4, 30, 40, " d"),
    });

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    // both steps stay within the 10 s window, so the anchor stays 0
    feed_ms(session, 3000);
    session.step();
    feed_ms(session, 3000);
    session.step();
    session.finish();

    int  finals = 0;
    bool has3   = false;
    int  partials = 0;
    int64_t last_rev = -1;
    for (const event & e : events) {
        if (e.kind == event_kind::final_segment) {
            ++finals;
            if (e.seg.text.find('c') != std::string::npos ||
                e.seg.text.find('d') != std::string::npos) {
                has3 = true;
            }
        } else if (e.kind == event_kind::partial_segment) {
            ++partials;
            REQUIRE(e.seg.revision > last_rev);
            last_rev = e.seg.revision;
        }
    }
    REQUIRE(finals >= 1);
    REQUIRE(!has3); // only ids 1 and 2 were in the stable prefix
    REQUIRE(partials >= 1);
}

// ---------- tail guard ----------

void test_tail_guard() {
    session_config cfg;
    cfg.tail_guard_ms = 1000;

    fake_transcriber tr;
    for (int i = 0; i < 2; ++i) {
        // local centiseconds: absolute t1 = 7100 ms, inside the 8 s window
        tr.scripts.push_back({
            make_tok(1, 1, 710, " a"),
            make_tok(2, 2, 710, " b"),
        });
    }

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    // two hypotheses within one window (anchor 0); the guard (fed - 1000 =
    // 7000 ms) sits just before the token end at 7100 ms
    feed_ms(session, 4000);
    session.step();
    feed_ms(session, 4000);
    session.step();

    for (const event & e : events) {
        REQUIRE(e.kind != event_kind::final_segment);
    }

    const int n_finish = session.finish();
    REQUIRE(n_finish >= 1);

    int finals = 0;
    for (const event & e : events) {
        if (e.kind == event_kind::final_segment) {
            ++finals;
            REQUIRE(e.seg.t1_ms <= 8000);
        }
    }
    REQUIRE(finals >= 1);
}

// ---------- monotonic timestamps + clamp accounting ----------

void test_monotonic_clamps() {
    session_config cfg;
    cfg.tail_guard_ms = 0;

    fake_transcriber tr;
    // first two hypotheses (same anchor 0): token with t1 < t0 is repaired
    tr.scripts.push_back({ make_tok(1, 300, 100, " a") });
    tr.scripts.push_back({ make_tok(1, 300, 100, " a") });
    // next two at anchor 0: a token ending before the previous final
    tr.scripts.push_back({ make_tok(2, 5, 15, " b") });
    tr.scripts.push_back({ make_tok(2, 5, 15, " b") });

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    feed_ms(session, 3000);
    session.step();
    feed_ms(session, 3000);
    session.step();
    feed_ms(session, 3000);
    session.step();
    feed_ms(session, 3000);
    session.step();
    session.finish();

    int64_t prev_t1 = 0;
    int     finals  = 0;
    bool    range_warn = false;
    for (const event & e : events) {
        if (e.kind == event_kind::timestamp_range) {
            range_warn = true;
        }
        if (e.kind == event_kind::final_segment) {
            REQUIRE(e.seg.t0_ms < e.seg.t1_ms);
            REQUIRE(e.seg.t0_ms >= prev_t1);
            prev_t1 = e.seg.t1_ms;
            REQUIRE(e.seg.t1_ms <= session.get_stats().fed_ms);
            ++finals;
        }
    }
    REQUIRE(finals >= 1);
    REQUIRE(session.get_stats().timestamp_clamps >= 1);
    REQUIRE(range_warn);
}

// ---------- overflow ----------

void test_overflow() {
    session_config cfg;
    cfg.capacity_ms = 60000;

    fake_transcriber tr;
    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    REQUIRE(push_status::ok ==
            session.push_audio(nullptr, 100)); // null pointer is a no-op

    feed_ms(session, 60000);
    const push_status st = [&]() {
        const int n = static_cast<int>(ms_samples(30000));
        std::vector<float> buf(n, 0.0f);
        return session.push_audio(buf.data(), n);
    }();

    REQUIRE(st == push_status::overflow);
    REQUIRE(session.get_stats().dropped_ms == 30000);

    int overflow_events = 0;
    for (const event & e : events) {
        if (e.kind == event_kind::audio_overflow) {
            ++overflow_events;
        }
    }
    REQUIRE(overflow_events == 1);
}

// ---------- retries / degradation / prompt reset ----------

void test_retries() {
    session_config cfg;
    cfg.max_retries = 2;
    cfg.tail_guard_ms = 0;

    fake_transcriber tr;
    tr.fail_flags = { true, true, true };
    tr.scripts.push_back({ make_tok(7, 10, 20, " x") });

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    feed_ms(session, 3000);

    session.step();
    REQUIRE(session.get_stats().retries == 1);
    session.step();
    REQUIRE(session.get_stats().retries == 2);

    const int64_t fed_before = session.get_stats().fed_ms;
    session.step(); // third consecutive failure -> degrade
    REQUIRE(session.get_stats().degraded);
    REQUIRE(session.get_stats().retries == 3);
    REQUIRE(session.get_stats().failed_ms == 3000);

    int fail_events = 0;
    for (const event & e : events) {
        if (e.kind == event_kind::transcribe_failed) {
            ++fail_events;
        }
    }
    REQUIRE(fail_events == 1);
    REQUIRE(fed_before == 3000);

    // new audio, one successful window: prompt must have been cleared
    feed_ms(session, 3000);
    session.step();
    REQUIRE(tr.last_prompt.empty());
}

// ---------- exceptions are treated as failures ----------

void test_exception_is_failure() {
    session_config cfg;
    cfg.max_retries = 0;

    fake_transcriber tr;
    tr.throw_next = true;

    stream_session session(cfg, tr, [](const event &) {});
    feed_ms(session, 3000);
    REQUIRE(session.step() >= 0); // must not propagate
    REQUIRE(session.get_stats().degraded);
}

// ---------- short window and empty result ----------

void test_short_and_empty() {
    session_config cfg;
    fake_transcriber tr;
    tr.scripts.push_back({}); // empty hypothesis

    std::vector<event> events;
    stream_session session(cfg, tr, [&](const event & e) {
        events.push_back(e);
    });

    feed_ms(session, 50);
    REQUIRE(session.step() == 0);
    REQUIRE(tr.calls == 0);

    feed_ms(session, 3000);
    REQUIRE(session.step() == 0); // empty result: no events
    REQUIRE(events.empty());
    REQUIRE(tr.calls == 1);
}

// ---------- finish() semantics ----------

void test_finish_repeat() {
    session_config cfg;
    fake_transcriber tr;
    tr.scripts.push_back({ make_tok(1, 10, 20, " a") });

    stream_session session(cfg, tr, [](const event &) {});

    REQUIRE(session.finish() == 0); // nothing buffered
    REQUIRE(session.get_stats().finish_calls == 1);
    REQUIRE(session.finish() == 0);
    REQUIRE(session.get_stats().finish_calls == 2);

    // writes after finish are rejected
    const int n = static_cast<int>(ms_samples(100));
    std::vector<float> buf(n, 0.0f);
    REQUIRE(session.push_audio(buf.data(), n) == push_status::closed);
}

// ---------- reset() behaves like a fresh session ----------

void test_reset() {
    session_config cfg;
    cfg.tail_guard_ms = 0;
    fake_transcriber tr;
    tr.scripts.push_back({ make_tok(1, 10, 20, " a") });
    tr.scripts.push_back({ make_tok(1, 10, 20, " a") });

    stream_session session(cfg, tr, [](const event &) {});
    feed_ms(session, 3000);
    session.step();
    session.finish();
    REQUIRE(session.get_stats().windows >= 1);

    session.reset();
    const stats s = session.get_stats();
    REQUIRE(s.windows == 0 && s.fed_ms == 0 && s.committed_ms == 0 &&
            s.finish_calls == 0 && s.degraded == false);

    // usable again
    feed_ms(session, 3000);
    REQUIRE(session.step() == 0);
}

// ---------- SPSC ring buffer under concurrent push/pop ----------

void test_ring_concurrent() {
    constexpr size_t N = 960000;
    audio_ring ring(1 << 16);
    REQUIRE(ring.capacity() == static_cast<size_t>(1 << 16));

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> dropped_total{0};

    std::thread producer([&]() {
        std::vector<float> block(173);
        size_t sent = 0;
        while (sent < N) {
            const size_t n = std::min(block.size(), N - sent);
            for (size_t i = 0; i < n; ++i) {
                block[i] = static_cast<float>((sent + i) % 4096);
            }
            dropped_total += ring.push(block.data(), n);
            sent += n;
            while (ring.size() > 32768) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true);
    });

    size_t       got      = 0;
    std::vector<float> out(4096);
    while (true) {
        const size_t n = ring.pop(out.data(), out.size());
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(static_cast<size_t>(out[i]) == got % 4096);
            ++got;
        }
        if (producer_done.load() && ring.size() == 0) {
            break;
        }
    }

    producer.join();
    REQUIRE(got == N);
    REQUIRE(dropped_total.load() == 0);
}

void test_ring_overwrite() {
    audio_ring ring(4); // rounded up to 4
    std::vector<float> in = {1, 2, 3, 4, 5, 6};
    const size_t dropped = ring.push(in.data(), in.size());
    REQUIRE(dropped == 2);
    REQUIRE(ring.size() == 4);

    float out[4] = {0};
    const size_t n = ring.pop(out, 4);
    REQUIRE(n == 4);
    REQUIRE(out[0] == 3 && out[1] == 4 && out[2] == 5 && out[3] == 6);
    REQUIRE(ring.pop(out, 4) == 0);
}

} // namespace

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };
    const test_case cases[] = {
        { "time_conversion",  test_time_conversion  },
        { "alignment",        test_alignment        },
        { "stable_prefix",    test_stable_prefix    },
        { "tail_guard",       test_tail_guard       },
        { "monotonic_clamps", test_monotonic_clamps },
        { "overflow",         test_overflow         },
        { "retries",          test_retries          },
        { "exception",        test_exception_is_failure },
        { "short_and_empty",  test_short_and_empty  },
        { "finish_repeat",    test_finish_repeat    },
        { "reset",            test_reset            },
        { "ring_concurrent",  test_ring_concurrent  },
        { "ring_overwrite",   test_ring_overwrite   },
    };

    int failed = 0;
    for (const test_case & tc : cases) {
        try {
            tc.fn();
            printf("ok - %s\n", tc.name);
        } catch (const std::exception & ex) {
            fprintf(stderr, "FAIL - %s: %s\n", tc.name, ex.what());
            ++failed;
        }
    }

    printf("%d checks, %d/%zu tests passed\n",
           g_checks, static_cast<int>(sizeof(cases) / sizeof(cases[0])) - failed,
           sizeof(cases) / sizeof(cases[0]));
    return failed == 0 ? 0 : 1;
}
