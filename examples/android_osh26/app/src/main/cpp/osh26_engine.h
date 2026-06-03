#pragma once

#include <functional>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <list>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace osh26 {

using TokenCallback = std::function<void(const std::string &)>;

struct GenerateOptions {
    int max_tokens = 128;
    float temperature = 0.6f;
    float top_p = 0.95f;
    uint32_t seed = 0xCAFE;
    bool enable_thinking = true;
};

struct GenerateResult {
    bool ok = false;
    bool cancelled = false;
    int decoded_tokens = 0;
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
    std::string finish_reason = "stop";
    std::string error;
    std::string text;
    std::string token_ids;
};

class ComputeBackend {
public:
    ComputeBackend();
    ~ComputeBackend();

    std::string load_model(const std::string & model_path);
    GenerateResult generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token);
    void configure_backend(const std::string & mode, int n_gpu_layers);
    void set_debug_correctness(bool enabled);
    void reset_cache();
    void cancel();
    void release();
    std::string stats_json() const;

private:
    enum class RequestState {
        queued,
        prefill,
        decode,
        done,
        cancelled,
        error,
    };

    struct PrefixCacheNode {
        std::unordered_map<llama_token, std::unique_ptr<PrefixCacheNode>> children;
        PrefixCacheNode * parent = nullptr;
        llama_token token = 0;
        bool terminal = false;
        size_t token_count = 0;
        size_t hit_count = 0;
        size_t request_count = 0;
        uint64_t last_used_tick = 0;
    };

    struct GenerationRequest {
        int id = 0;
        std::string prompt;
        GenerateOptions options;
        TokenCallback on_token;
        std::vector<llama_token> prompt_tokens;
        GenerateResult result;
        size_t prompt_tokens_total = 0;
        size_t reusable_prefix_tokens = 0;
        size_t queue_position = 0;
        RequestState state = RequestState::queued;
        bool started = false;
        bool completed = false;
        bool cancelled = false;
        bool rejected = false;
        double queue_wait_ms = 0.0;
        double prompt_build_ms = 0.0;
        double tokenize_ms = 0.0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        std::chrono::steady_clock::time_point submitted_at;
        std::chrono::steady_clock::time_point started_at;
        std::chrono::steady_clock::time_point finished_at;
        std::mutex mutex;
        std::condition_variable cv;
    };

    std::string build_prompt(const std::string & user_prompt, bool enable_thinking) const;
    std::string build_prompt_prefix() const;
    bool warm_prefix_cache_locked();
    bool prompt_has_cached_prefix(const std::vector<llama_token> & prompt_tokens) const;
    size_t common_prefix_length(const std::vector<llama_token> & lhs, const std::vector<llama_token> & rhs) const;
    std::shared_ptr<GenerationRequest> enqueue_request(
        const std::string & prompt,
        const GenerateOptions & options,
        const TokenCallback & on_token);
    void ensure_worker_started_locked();
    void worker_loop();
    std::shared_ptr<GenerationRequest> pick_next_request_locked();
    void complete_request(const std::shared_ptr<GenerationRequest> & request);
    void fail_queued_requests_locked(const std::string & error_message, bool cancelled);
    void reset_cache_locked(bool clear_prefix_state);
    bool process_request(const std::shared_ptr<GenerationRequest> & request);
    bool process_request_cpu(const std::shared_ptr<GenerationRequest> & request, const llama_vocab * vocab, llama_sampler * sampler);
    bool process_request_gpu(const std::shared_ptr<GenerationRequest> & request, const llama_vocab * vocab, llama_sampler * sampler);
    bool prefill_prompt_gpu(const std::shared_ptr<GenerationRequest> & request, const llama_token * tokens, int n_tokens, int start_pos, bool * cancelled);
    bool prefill_prompt_cpu(const std::shared_ptr<GenerationRequest> & request, llama_batch & batch, bool * cancelled);
    void insert_prefix_cache_locked(const std::vector<llama_token> & prompt_tokens);
    PrefixCacheNode * ensure_prefix_node_locked(const std::vector<llama_token> & tokens);
    PrefixCacheNode * find_prefix_node_locked(const std::vector<llama_token> & tokens) const;
    void touch_prefix_node_locked(PrefixCacheNode * node);
    void evict_prefix_cache_locked();
    size_t prefix_cache_entry_count_locked() const;
    size_t prefix_cache_token_count_locked() const;
    double prefix_cache_fragmentation_locked() const;

    mutable std::mutex mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_thread_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    std::string model_path_;
    std::atomic_bool cancel_requested_{false};
    bool worker_started_ = false;
    bool shutdown_requested_ = false;
    bool running_ = false;
    size_t next_request_id_ = 1;
    size_t next_queue_position_ = 1;
    std::deque<std::shared_ptr<GenerationRequest>> request_queue_;
    std::shared_ptr<GenerationRequest> active_request_;
    std::vector<llama_token> active_prompt_tokens_;
    std::vector<llama_token> boot_prefix_tokens_;
    PrefixCacheNode prefix_cache_root_;
    std::list<PrefixCacheNode *> prefix_cache_lru_;
    size_t prefix_cache_entry_count_ = 0;
    size_t prefix_cache_token_count_ = 0;
    size_t prefix_cache_evictions_ = 0;
    size_t prefix_cache_hits_ = 0;
    size_t prefix_cache_misses_ = 0;
    size_t prefix_cache_reuse_tokens_ = 0;
    size_t prefix_cache_block_reuse_ = 0;
    size_t prefix_cache_block_size_ = 16;
    size_t max_prefix_cache_entries_ = 32;
    size_t max_prefix_cache_tokens_ = 256;
    size_t max_pending_requests_ = 8;
    size_t queue_depth_peak_ = 0;
    size_t total_submitted_requests_ = 0;
    size_t total_completed_requests_ = 0;
    size_t total_cancelled_requests_ = 0;
    size_t total_rejected_requests_ = 0;
    size_t total_prefill_chunks_ = 0;
    size_t total_decode_steps_ = 0;
    double total_queue_wait_ms_ = 0.0;
    double total_prefill_ms_ = 0.0;
    double total_decode_ms_ = 0.0;
    int last_decoded_tokens_ = 0;
    double last_ttft_ms_ = 0.0;
    double last_tokens_per_second_ = 0.0;
    std::string last_finish_reason_ = "stop";
    std::string last_error_;
    std::string active_backend_ = "llama.cpp CPU";
    std::string available_devices_;
    std::string requested_backend_ = "vulkan";
    int requested_gpu_layers_ = -1;
    bool debug_correctness_ = false;
    std::string last_token_ids_;
    bool prefix_cache_valid_ = false;
    int prefix_cached_pos_ = 0;
    std::vector<llama_token> prefix_tokens_;
    double last_prompt_build_ms_ = 0.0;
    double last_tokenize_ms_ = 0.0;
    bool last_prefix_cache_hit_ = false;
    int last_prefix_tokens_ = 0;
    double last_queue_wait_ms_ = 0.0;
    int last_user_prefill_tokens_ = 0;
    int last_prefill_forward_count_ = 0;
    int last_prefill_skipped_lm_head_count_ = 0;
    int last_prefill_chunk_size_ = 0;
    int last_prefill_chunk_count_ = 0;
    int last_prefill_logits_chunks_ = 0;
    double last_user_prefill_ms_ = 0.0;
    double last_first_decode_ms_ = 0.0;
    size_t last_reusable_prefix_tokens_ = 0;
    size_t last_cached_prefix_entries_ = 0;
    bool last_logits_sanity_ok_ = true;
    int last_logits_low_id_streak_ = 0;
    std::string last_logits_sanity_reason_;
};

class SchedulerLite {
public:
    std::string load_model(const std::string & model_path);
    GenerateResult generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token);
    void configure_backend(const std::string & mode, int n_gpu_layers);
    void set_debug_correctness(bool enabled);
    void reset_cache();
    void cancel();
    void release();
    std::string stats_json() const;

private:
    ComputeBackend backend_;
};

SchedulerLite & engine();
void init_llama_backend();

} // namespace osh26
