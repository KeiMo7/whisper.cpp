#include "stream-session.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace whisper_stream {

int64_t samples_to_ms(int64_t n) {
    return n * 1000 / WHISPER_STREAM_SAMPLE_RATE;
}

int64_t cs_to_ms(int64_t cs) {
    return cs * 10;
}

stream_session::stream_session(const session_config & cfg,
                               stream_transcriber & transcriber,
                               std::function<void(const event &)> event_sink)
    : cfg_(cfg),
      tr_(transcriber),
      sink_(std::move(event_sink)) {}

int64_t stream_session::ms_to_samples(int64_t ms) const {
    return ms * WHISPER_STREAM_SAMPLE_RATE / 1000;
}

int64_t stream_session::fed_ms() const {
    return samples_to_ms(fed_smpl_);
}

int64_t stream_session::buffer_start_ms() const {
    return samples_to_ms(start_smpl_);
}

void stream_session::refresh_lag() {
    st_.fed_ms       = fed_ms();
    st_.lag_ms       = st_.fed_ms - st_.committed_ms;
}

void stream_session::emit_warning(event_kind code, int64_t detail_ms) {
    event e;
    e.kind           = code;
    e.warn.code      = code;
    e.warn.detail_ms = detail_ms;
    ++delivered_;
    sink_(e);
}

void stream_session::invalidate_hypothesis() {
    have_prev_hyp_  = false;
    prev_anchor_ms_ = -1;
    prev_hyp_.clear();
}

void stream_session::trim_audio_to_start() {
    // audio_ holds [origin_smpl_, fed_smpl_); discard samples before the
    // window start. start_smpl_ never moves backwards, origin_smpl_ may lag.
    if (start_smpl_ <= origin_smpl_) {
        return;
    }
    const int64_t drop = std::min(start_smpl_ - origin_smpl_,
                                  static_cast<int64_t>(audio_.size()));
    audio_.erase(audio_.begin(),
                 audio_.begin() + static_cast<std::ptrdiff_t>(drop));
    origin_smpl_ += drop;

    // if the vector was shorter than expected, align the origin exactly so
    // that audio_.size() == fed_smpl_ - origin_smpl_ stays invariant
    if (origin_smpl_ < start_smpl_) {
        origin_smpl_ = start_smpl_;
    }
}

push_status stream_session::push_audio(const float * samples, int n_samples) {
    if (finished_) {
        return push_status::closed;
    }
    if (samples == nullptr || n_samples <= 0) {
        return push_status::ok;
    }

    const int64_t cap = ms_to_samples(cfg_.capacity_ms);
    const int64_t n   = n_samples;

    push_status status = push_status::ok;

    if (static_cast<int64_t>(audio_.size()) + n > cap) {
        // keep the cap newest samples; drop only what does not fit
        const int64_t keep_new = std::min(n, cap);
        const int64_t room     = cap - keep_new;

        int64_t n_drop = 0;
        if (static_cast<int64_t>(audio_.size()) > room) {
            n_drop = static_cast<int64_t>(audio_.size()) - room;
            audio_.erase(audio_.begin(),
                         audio_.begin() + static_cast<std::ptrdiff_t>(n_drop));
        }
        if (n > keep_new) {
            n_drop += n - keep_new;
        }

        origin_smpl_ += n_drop;
        start_smpl_   = std::max(start_smpl_, origin_smpl_);

        st_.dropped_ms += samples_to_ms(n_drop);
        emit_warning(event_kind::audio_overflow, samples_to_ms(n_drop));

        prompt_tokens_.clear();
        invalidate_hypothesis();
        status = push_status::overflow;

        audio_.insert(audio_.end(), samples + (n - keep_new), samples + n);
    } else {
        audio_.insert(audio_.end(), samples, samples + n);
    }

    fed_smpl_ += n;
    refresh_lag();
    return status;
}

void stream_session::handle_failure(int64_t span_ms, bool finishing) {
    ++st_.retries;

    if (st_.retries <= cfg_.max_retries) {
        // keep window start, watermarks and prompt; the next step retries
        return;
    }

    emit_warning(event_kind::transcribe_failed, span_ms);

    st_.degraded   = true;
    st_.failed_ms += span_ms;
    if (finishing) {
        st_.truncated = true;
    }

    // skip the whole pending window
    start_smpl_  = fed_smpl_;
    origin_smpl_ = fed_smpl_;
    audio_.clear();

    prompt_tokens_.clear();
    invalidate_hypothesis();
    silence_ms_ = 0;
}

void stream_session::clamp_window_tokens(std::vector<hyp_token> & toks,
                                         int64_t span_ms,
                                         bool & range_warned) {
    const int64_t span_cs = span_ms / 10;
    int64_t       fixes   = 0;

    for (hyp_token & tok : toks) {
        int64_t t0 = tok.t0_ms; // local centiseconds during this pass
        int64_t t1 = tok.t1_ms;
        bool    touched = false;

        if (t0 < 0) {
            t0 = 0;
            touched = true;
        }
        if (t1 < 0) {
            t1 = 0;
            touched = true;
        }
        if (t0 > span_cs) {
            t0 = span_cs;
            touched = true;
        }
        if (t1 > span_cs) {
            t1 = span_cs;
            touched = true;
        }
        if (t1 < t0) {
            t1 = std::min(span_cs, t0 + 1);
            if (t1 < t0) {
                t0 = std::max<int64_t>(0, span_cs - 1);
                t1 = span_cs;
            }
            touched = true;
        }

        if (touched) {
            ++fixes;
        }
        tok.t0_ms = t0;
        tok.t1_ms = t1;
    }

    if (fixes > 0 && !range_warned) {
        range_warned = true;
        emit_warning(event_kind::timestamp_range, fixes);
    }
    st_.timestamp_clamps += fixes;
}

void stream_session::append_to_segment(const hyp_token & tok) {
    if (!seg_open_) {
        seg_open_     = true;
        seg_t0_ms_    = tok.t0_ms;
        seg_t1_ms_    = tok.t1_ms;
        seg_first_ms_ = tok.t0_ms;
        seg_text_     = tok.text;
        // strip leading whitespace only at the start of a segment
        size_t pos = 0;
        while (pos < seg_text_.size() &&
               (seg_text_[pos] == ' ' || seg_text_[pos] == '\t')) {
            ++pos;
        }
        seg_text_.erase(0, pos);
        return;
    }

    seg_text_ += tok.text;
    seg_t1_ms_ = tok.t1_ms;
}

void stream_session::close_segment(int64_t revision, bool final_segment) {
    if (!seg_open_) {
        return;
    }
    seg_open_ = false;

    bool blank = true;
    for (char c : seg_text_) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            blank = false;
            break;
        }
    }
    if (blank) {
        seg_text_.clear();
        return;
    }

    int64_t t0 = seg_t0_ms_;
    int64_t t1 = seg_t1_ms_;

    if (final_segment) {
        bool clamped = false;
        if (t0 < last_final_t1_ms_) {
            t0 = last_final_t1_ms_;
            clamped = true;
        }
        if (t1 > st_.fed_ms) {
            t1 = st_.fed_ms;
            clamped = true;
        }
        if (t0 >= t1) {
            t0 = last_final_t1_ms_;
            t1 = t0 + 10;
            clamped = true;
        }
        if (clamped) {
            ++st_.timestamp_clamps;
            if (!range_warned_) {
                range_warned_ = true;
                emit_warning(event_kind::timestamp_range, 1);
            }
        }
        last_final_t1_ms_ = t1;
    }

    event e;
    e.kind = final_segment ? event_kind::final_segment
                           : event_kind::partial_segment;
    e.seg.t0_ms    = t0;
    e.seg.t1_ms    = t1;
    e.seg.text     = seg_text_;
    e.seg.final    = final_segment;
    e.seg.revision = revision;

    seg_text_.clear();
    ++delivered_;
    sink_(e);
}

void stream_session::emit_partial(int64_t revision) {
    if (prev_hyp_.empty()) {
        return;
    }

    int64_t      t0 = prev_hyp_.front().t0_ms;
    int64_t      t1 = prev_hyp_.back().t1_ms;
    std::string  text;
    for (const hyp_token & tok : prev_hyp_) {
        text += tok.text;
    }

    bool blank = true;
    for (char c : text) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            blank = false;
            break;
        }
    }
    if (blank) {
        return;
    }

    size_t pos = 0;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
    }
    text.erase(0, pos);

    event e;
    e.kind         = event_kind::partial_segment;
    e.seg.t0_ms    = t0;
    e.seg.t1_ms    = t1;
    e.seg.text     = text;
    e.seg.final    = false;
    e.seg.revision = revision;
    ++delivered_;
    sink_(e);
}

void stream_session::rebuild_prompt() {
    const int32_t eot = tr_.eot_id();
    const int32_t ctx = tr_.text_ctx();

    int32_t limit = cfg_.prompt_max_tokens;
    if (ctx > 0) {
        limit = std::min(limit, ctx / 2);
    }
    if (limit <= 0) {
        prompt_tokens_.clear();
        return;
    }

    std::vector<int32_t> next;
    const int64_t avail = static_cast<int64_t>(committed_ids_.size());
    const int64_t first = std::max<int64_t>(0, avail - limit);

    for (int64_t i = first; i < avail; ++i) {
        const int32_t id = committed_ids_[static_cast<size_t>(i)];
        if (id < eot) {
            continue; // drop special tokens
        }
        next.push_back(id);
    }
    prompt_tokens_.swap(next);
}

void stream_session::commit_tokens(std::vector<hyp_token> & toks, bool finishing) {
    if (toks.empty()) {
        return;
    }

    const int64_t base_ms  = buffer_start_ms();
    const int64_t fed      = fed_ms();
    bool          prev_turn = false;

    for (size_t i = 0; i < toks.size(); ++i) {
        hyp_token tok = toks[i];

        // local centiseconds -> absolute session milliseconds
        tok.t0_ms = base_ms + cs_to_ms(tok.t0_ms);
        tok.t1_ms = base_ms + cs_to_ms(tok.t1_ms);

        if (finishing) {
            if (tok.t1_ms > fed) {
                tok.t1_ms = fed;
            }
            if (tok.t0_ms > tok.t1_ms) {
                tok.t0_ms = tok.t1_ms;
            }
            if (tok.t1_smpl > fed_smpl_) {
                tok.t1_smpl = fed_smpl_;
            }
        }

        // watermark deduplication
        if (tok.t1_ms <= st_.committed_ms) {
            continue;
        }
        if (tok.t0_ms < st_.committed_ms) {
            tok.t0_ms = st_.committed_ms;
        }
        if (tok.t1_ms < tok.t0_ms) {
            tok.t1_ms = tok.t0_ms;
        }

        bool split = false;
        if (seg_open_) {
            const int64_t gap = tok.t0_ms - seg_t1_ms_;
            if (gap >= cfg_.segment_gap_ms ||
                prev_turn ||
                (tok.t1_ms - seg_first_ms_) >= cfg_.max_segment_ms) {
                split = true;
            }
        }
        if (split) {
            close_segment(st_.windows, true);
        }

        append_to_segment(tok);

        st_.committed_ms = std::max(st_.committed_ms, tok.t1_ms);
        committed_smpl_  = std::max(committed_smpl_, tok.t1_smpl);
        committed_ids_.push_back(tok.id);
        prev_turn = tok.speaker_turn_next;
    }
}

int stream_session::run_window(bool finishing) {
    delivered_    = 0;
    range_warned_ = false;

    const int64_t start_ms = buffer_start_ms();
    const int64_t span_ms  = fed_ms() - start_ms;
    const int     n        = static_cast<int>(audio_.size());

    std::vector<token_time> raw;
    std::string             err;
    bool                    ok = false;

    try {
        ok = tr_.transcribe(audio_.data(), n, start_ms,
                            prompt_tokens_, raw, err);
    } catch (const std::exception &) {
        ok = false;
    }

    if (!ok) {
        handle_failure(span_ms, finishing);
        if (finishing) {
            close_segment(st_.windows, true);
        }
        return delivered_;
    }

    st_.retries = 0;
    ++st_.windows;

    std::vector<hyp_token> hyp;
    hyp.reserve(raw.size());
    for (const token_time & t : raw) {
        hyp_token tok;
        tok.id                = t.id;
        tok.t0_ms             = t.t0_cs; // local cs until commit_tokens()
        tok.t1_ms             = t.t1_cs;
        tok.t1_smpl           = start_smpl_ +
                                cs_to_ms(t.t1_cs) * WHISPER_STREAM_SAMPLE_RATE / 1000;
        tok.text              = t.text;
        tok.speaker_turn_next = t.speaker_turn_next;
        hyp.push_back(tok);
    }

    clamp_window_tokens(hyp, span_ms, range_warned_);

    if (hyp.empty()) {
        // nothing decoded: slide the window forward instead of re-decoding
        // the same silence forever
        silence_ms_ += cfg_.step_ms;
        if (cfg_.vad && silence_ms_ >= cfg_.prompt_reset_silence_ms) {
            prompt_tokens_.clear();
            silence_ms_ = 0;
        }

        start_smpl_ = std::max(start_smpl_,
                               std::min(fed_smpl_,
                                        start_smpl_ + ms_to_samples(cfg_.step_ms)));
        trim_audio_to_start();
        invalidate_hypothesis();
        if (finishing) {
            close_segment(st_.windows, true);
        }
        refresh_lag();
        return delivered_;
    }
    silence_ms_ = 0;

    // stable prefix over the uncommitted tails of consecutive hypotheses
    size_t commit_cnt = 0;

    if (!finishing && have_prev_hyp_ && prev_anchor_ms_ == start_ms) {
        size_t lcp = 0;
        const size_t max_lcp = std::min(prev_hyp_.size(), hyp.size());
        while (lcp < max_lcp && prev_hyp_[lcp].id == hyp[lcp].id) {
            ++lcp;
        }

        const int64_t guard = fed_ms() - cfg_.tail_guard_ms;
        for (size_t i = 0; i < lcp; ++i) {
            const int64_t t1_abs = start_ms + cs_to_ms(hyp[i].t1_ms);
            if (t1_abs > guard) {
                break;
            }
            ++commit_cnt;
        }
    }

    std::vector<hyp_token> committing;
    if (finishing) {
        committing = hyp;
    } else {
        committing.assign(hyp.begin(), hyp.begin() + commit_cnt);
    }

    commit_tokens(committing, finishing);

    if (!finishing) {
        // only the uncommitted tail participates in the next comparison
        prev_hyp_.assign(hyp.begin() + commit_cnt, hyp.end());
        have_prev_hyp_  = true;
        prev_anchor_ms_ = start_ms;
    } else {
        prev_hyp_.clear();
        have_prev_hyp_ = false;
    }

    if (!committing.empty()) {
        rebuild_prompt();

        // advance the window to (committed - keep), entirely in sample
        // space so no millisecond truncation can move the start backwards
        const int64_t keep_smpl = ms_to_samples(cfg_.keep_ms);
        const int64_t new_start_smpl =
            std::max(start_smpl_, committed_smpl_ - keep_smpl);
        if (new_start_smpl > start_smpl_) {
            start_smpl_ = new_start_smpl;
            trim_audio_to_start();
            if (!finishing) {
                invalidate_hypothesis();
            }
        }
    }

    if (finishing) {
        close_segment(st_.windows, true);
    } else {
        // every committed batch is emitted as a final segment; the segment
        // is only kept open across batches when the next token continues it
        if (!committing.empty()) {
            close_segment(st_.windows, true);
        }
        if (cfg_.emit_partials) {
            // partial times are absolute; convert the tail for display only
            std::vector<hyp_token> display = prev_hyp_;
            for (hyp_token & tok : display) {
                tok.t0_ms = start_ms + cs_to_ms(tok.t0_ms);
                tok.t1_ms = start_ms + cs_to_ms(tok.t1_ms);
            }
            std::vector<hyp_token> saved = std::move(prev_hyp_);
            prev_hyp_ = std::move(display);
            emit_partial(st_.windows);
            prev_hyp_ = std::move(saved);
        }
    }

    refresh_lag();
    return delivered_;
}

int stream_session::step() {
    if (finished_) {
        return 0;
    }

    // compare in samples: a window whose millisecond truncation looks
    // exactly at the limit may still be a few samples longer, which must
    // not trigger clipping.
    const int64_t span_smpl = fed_smpl_ - start_smpl_;
    if (samples_to_ms(span_smpl) < 100) {
        return 0;
    }

    const int64_t window_smpl = ms_to_samples(cfg_.window_ms);
    if (span_smpl > window_smpl) {
        start_smpl_ = std::max(start_smpl_, fed_smpl_ - window_smpl);
        trim_audio_to_start();
        invalidate_hypothesis();
    }

    return run_window(false);
}

int stream_session::finish() {
    if (finished_) {
        ++st_.finish_calls;
        return 0;
    }
    finished_ = true;
    ++st_.finish_calls;

    delivered_    = 0;
    range_warned_ = false;

    const int64_t span_ms = fed_ms() - buffer_start_ms();

    // Flush an already available hypothesis directly; only run one last
    // transcription when nothing was decoded for the current window.
    if (have_prev_hyp_ &&
        (span_ms < 100 || prev_anchor_ms_ == buffer_start_ms())) {
        commit_tokens(prev_hyp_, true);
        prev_hyp_.clear();
        have_prev_hyp_ = false;
        close_segment(st_.windows, true);
        refresh_lag();
        return delivered_;
    }

    if (span_ms >= 100) {
        return run_window(true);
    }

    close_segment(st_.windows, true);
    refresh_lag();
    return delivered_;
}

void stream_session::reset() {
    st_                 = stats{};
    audio_.clear();
    fed_smpl_           = 0;
    origin_smpl_        = 0;
    start_smpl_         = 0;
    committed_smpl_     = 0;
    prev_hyp_.clear();
    have_prev_hyp_      = false;
    prev_anchor_ms_     = -1;
    prompt_tokens_.clear();
    committed_ids_.clear();
    silence_ms_         = 0;
    finished_           = false;
    seg_open_           = false;
    seg_t0_ms_          = 0;
    seg_t1_ms_          = 0;
    seg_first_ms_       = 0;
    seg_text_.clear();
    last_final_t1_ms_   = 0;
    delivered_          = 0;
    range_warned_       = false;
}

} // namespace whisper_stream
