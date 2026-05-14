#pragma once

#include <functional>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

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
};

class ComputeBackend {
public:
    ComputeBackend();
    ~ComputeBackend();

    std::string load_model(const std::string & model_path);
    GenerateResult generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token);
    void cancel();
    void release();
    std::string stats_json() const;

private:
    std::string build_prompt(const std::string & user_prompt, bool enable_thinking) const;

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
};

class SchedulerLite {
public:
    std::string load_model(const std::string & model_path);
    GenerateResult generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token);
    void cancel();
    void release();
    std::string stats_json() const;

private:
    ComputeBackend backend_;
};

SchedulerLite & engine();
void init_llama_backend();

} // namespace osh26
