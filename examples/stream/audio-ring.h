#pragma once

// Single-producer, single-consumer lock-free ring buffer of audio samples.
//
// The producer (audio capture callback) only calls push(); the consumer
// (processing thread) only calls pop()/size(). The two sides share no
// mutable state besides the two atomic positions below, which are
// synchronized exclusively with acquire/release ordering.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace whisper_stream {

class audio_ring {
public:
    // capacity_samples is rounded up to the next power of two.
    explicit audio_ring(size_t capacity_samples)
        : mask_(round_pow2(capacity_samples) - 1),
          buf_(mask_ + 1) {}

    audio_ring(const audio_ring &)             = delete;
    audio_ring & operator=(const audio_ring &) = delete;

    // Append up to n samples. If the buffer cannot hold all of them, the
    // oldest unread samples are overwritten first and the number of
    // overwritten samples is returned. Only the producer may call this.
    size_t push(const float * samples, size_t n) {
        if (samples == nullptr || n == 0) {
            return 0;
        }

        const size_t cap    = mask_ + 1;
        const size_t head   = write_pos_.load(std::memory_order_relaxed);
        const size_t tail   = read_pos_.load(std::memory_order_acquire);
        const size_t filled = head - tail;

        size_t dropped = 0;
        if (n + filled > cap) {
            dropped = n + filled - cap;
            // advance the consumer position past the overwritten samples;
            // the consumer may concurrently read the old value, hence RMW.
            read_pos_.fetch_add(dropped, std::memory_order_acq_rel);
        }

        size_t new_head = head;
        for (size_t i = 0; i < n; ++i) {
            buf_[new_head & mask_] = samples[i];
            ++new_head;
        }
        write_pos_.store(new_head, std::memory_order_release);
        return dropped;
    }

    // Remove at most max_out samples into out and return how many were
    // actually copied. Only the consumer may call this.
    size_t pop(float * out, size_t max_out) {
        if (out == nullptr || max_out == 0) {
            return 0;
        }

        const size_t head = write_pos_.load(std::memory_order_acquire);
        size_t       tail = read_pos_.load(std::memory_order_relaxed);

        const size_t avail = head - tail;
        const size_t n     = avail < max_out ? avail : max_out;

        for (size_t i = 0; i < n; ++i) {
            out[i] = buf_[tail & mask_];
            ++tail;
        }
        read_pos_.store(tail, std::memory_order_release);
        return n;
    }

    // Number of unread samples. Consumer-side estimate; only the consumer
    // is expected to rely on it.
    size_t size() const {
        const size_t head = write_pos_.load(std::memory_order_acquire);
        const size_t tail = read_pos_.load(std::memory_order_acquire);
        return head - tail;
    }

    size_t capacity() const {
        return mask_ + 1;
    }

private:
    static size_t round_pow2(size_t x) {
        if (x < 2) {
            return 2;
        }
        size_t p = 1;
        while (p < x) {
            p <<= 1;
        }
        return p;
    }

    const size_t       mask_;
    std::vector<float> buf_;

    std::atomic<size_t> read_pos_  {0};
    std::atomic<size_t> write_pos_ {0};
};

} // namespace whisper_stream
