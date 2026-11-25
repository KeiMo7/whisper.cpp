#pragma once

// SDL/microphone independent streaming transcription session.
//
// A stream_session owns the unprocessed audio buffer and all scheduling
// state: window selection, calls into a stream_transcriber, stable-prefix
// detection, segment assembly and timestamp alignment. All times are
// integer milliseconds (or centiseconds at the transcriber boundary) and
// are derived solely from sample counts; no system clock is ever read.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "whisper.h"

namespace whisper_stream {

constexpr int WHISPER_STREAM_SAMPLE_RATE = WHISPER_SAMPLE_RATE; // 16000

// n * 1000 / 16000, 64-bit, truncated towards zero.
int64_t samples_to_ms(int64_t n);

// centiseconds -> milliseconds.
int64_t cs_to_ms(int64_t cs);

enum class push_status {
    ok,        // all samples were buffered
    overflow,  // the oldest unprocessed samples had to be dropped
    closed,    // finish() was already called
};

enum class event_kind {
    final_segment,
    partial_segment,
    audio_overflow,
    transcribe_failed,
    timestamp_range,
};

// One transcribed token. t0_cs/t1_cs are local to the audio handed to the
// transcriber (i.e. relative to the window start), in centiseconds.
struct token_time {
    int32_t     id = 0;
    int64_t     t0_cs = 0;
    int64_t     t1_cs = 0;
    std::string text;
    bool        speaker_turn_next = false;
};

struct segment {
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    std::string text;
    bool        final = true;
    int64_t     revision = 0; // stats.windows at the time of emission
};

struct warning {
    event_kind code = event_kind::transcribe_failed;
    int64_t     detail_ms = 0;
};

struct event {
    event_kind kind = event_kind::final_segment;
    segment    seg;
    warning    warn;
};

struct session_config {
    int64_t step_ms                = 3000;
    int64_t window_ms              = 10000;
    int64_t keep_ms                = 200;
    int64_t tail_guard_ms          = 1000;
    int64_t segment_gap_ms         = 500;
    int64_t max_segment_ms         = 15000;
    int64_t capacity_ms            = 60000;
    int32_t prompt_max_tokens      = 224;
    int64_t prompt_reset_silence_ms = 2000;
    int32_t max_retries            = 2;
    bool    emit_partials          = false;
    bool    vad                    = false;
    int64_t vad_min_silence_ms     = 300;
};

struct stats {
    int64_t windows           = 0; // number of actual transcribe() calls
    int64_t fed_ms            = 0; // total audio accepted via push_audio()
    int64_t committed_ms      = 0; // max t1_ms of committed tokens
    int64_t lag_ms            = 0; // fed_ms - committed_ms
    int64_t dropped_ms        = 0; // audio lost to overflow
    int64_t failed_ms         = 0; // audio skipped after exhausting retries
    int64_t timestamp_clamps  = 0;
    int32_t retries           = 0; // consecutive failures in the current window
    bool    degraded          = false;
    int32_t finish_calls      = 0;
    bool    truncated         = false;
};

// Abstract transcription backend. Implemented against whisper_full() by
// whisper_transcriber in stream-whisper.cpp, and by scripted fakes in tests.
class stream_transcriber {
public:
    virtual ~stream_transcriber() = default;

    // Transcribe n_samples starting at window_start_ms (global session time).
    // Returning false means the attempt failed with err describing the error;
    // the session keeps the window and retries. Thrown std::exceptions are
    // treated identically to a false return.
    virtual bool transcribe(const float * samples, int n_samples,
                            int64_t window_start_ms,
                            const std::vector<int32_t> & prompt_tokens,
                            std::vector<token_time> & out,
                            std::string & err) = 0;

    // First special token id; prompt tokens with smaller ids are kept.
    // Defaults to 0, i.e. nothing is filtered.
    virtual int32_t eot_id() const { return 0; }

    // Text context size, used to cap the prompt at n_text_ctx()/2.
    // A non-positive value disables the extra cap.
    virtual int32_t text_ctx() const { return 0; }
};

class stream_session {
public:
    stream_session(const session_config & cfg,
                   stream_transcriber & transcriber,
                   std::function<void(const event &)> event_sink);

    push_status push_audio(const float * samples, int n_samples);

    // Process one scheduling step. Returns the number of events delivered
    // to the sink during this call. Never throws.
    int step();

    // Flush the remaining audio, ignoring the tail guard. Idempotent;
    // subsequent calls deliver nothing. Never throws.
    int finish();

    const stats & get_stats() const { return st_; }

    // Return to the exact state of a freshly constructed session.
    void reset();

private:
    struct hyp_token {
        int32_t     id = 0;
        int64_t     t0_ms = 0; // local cs during a window, absolute ms after commit
        int64_t     t1_ms = 0;
        int64_t     t1_smpl = 0; // absolute sample index of the token end
        std::string text;
        bool        speaker_turn_next = false;
    };

    int64_t ms_to_samples(int64_t ms) const;
    int64_t fed_ms() const;
    int64_t buffer_start_ms() const;
    void    refresh_lag();
    void    emit_warning(event_kind code, int64_t detail_ms);
    void    trim_audio_to_start();
    void    invalidate_hypothesis();
    void    handle_failure(int64_t span_ms, bool finishing);
    void    clamp_window_tokens(std::vector<hyp_token> & toks, int64_t span_ms,
                                bool & range_warned);
    void    commit_tokens(std::vector<hyp_token> & toks, bool finishing);
    void    append_to_segment(const hyp_token & tok);
    void    close_segment(int64_t revision, bool final_segment);
    void    emit_partial(int64_t revision);
    void    rebuild_prompt();
    int     run_window(bool finishing);

    session_config                     cfg_;
    stream_transcriber &               tr_;
    std::function<void(const event &)> sink_;

    stats                 st_;
    std::vector<float>    audio_;            // unprocessed samples
    int64_t               fed_smpl_      = 0; // total samples accepted
    int64_t               origin_smpl_   = 0; // sample index of audio_[0]
    int64_t               start_smpl_    = 0; // sample index of window start
    int64_t               committed_smpl_ = 0; // sample at committed watermark
    std::vector<hyp_token> prev_hyp_;
    bool                  have_prev_hyp_    = false;
    int64_t               prev_anchor_ms_   = -1; // buffer_start of prev_hyp_
    std::vector<int32_t>  prompt_tokens_;
    std::vector<int32_t>  committed_ids_;
    int64_t               silence_ms_       = 0;
    bool                  finished_         = false;
    int                   delivered_        = 0; // events of current call
    bool                  range_warned_     = false;

    // open segment being assembled
    bool                  seg_open_         = false;
    int64_t               seg_t0_ms_        = 0;
    int64_t               seg_t1_ms_        = 0;
    int64_t               seg_first_ms_     = 0; // timestamp of first token
    std::string           seg_text_;
    int64_t               last_final_t1_ms_ = 0;
};

} // namespace whisper_stream
