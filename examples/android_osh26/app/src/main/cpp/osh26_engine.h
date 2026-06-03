#pragma once

#include <functional>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
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
    std::string build_prompt(const std::string & user_prompt, bool enable_thinking) const;
    std::string build_prompt_prefix() const;
    bool warm_prefix_cache_locked();
    bool prompt_has_cached_prefix(const std::vector<llama_token> & prompt_tokens) const;
    void reset_cache_locked();

    mutable std::mutex mutex_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    std::string model_path_;
    std::atomic_bool cancel_requested_{false};
    bool running_ = false;
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
    int last_user_prefill_tokens_ = 0;
    int last_prefill_forward_count_ = 0;
    int last_prefill_skipped_lm_head_count_ = 0;
    double last_user_prefill_ms_ = 0.0;
    double last_first_decode_ms_ = 0.0;
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
