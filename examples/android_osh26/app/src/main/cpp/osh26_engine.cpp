#include "osh26_engine.h"

#include <android/log.h>
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace osh26 {
namespace {

constexpr int kDefaultContextSize = 2048;
constexpr int kDefaultBatchSize = 512;
constexpr int kDefaultMaxSeq = 4;

std::once_flag g_backend_once;

void android_llama_log(ggml_log_level level, const char * text, void *) {
    int priority = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_ERROR) {
        priority = ANDROID_LOG_ERROR;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        priority = ANDROID_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_DEBUG) {
        priority = ANDROID_LOG_DEBUG;
    }
    __android_log_write(priority, "OSH26Llama", text == nullptr ? "" : text);
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:   out << c; break;
        }
    }
    return out.str();
}

std::string describe_file(const std::string & path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        std::ostringstream out;
        out << "stat failed: " << std::strerror(errno) << " (" << errno << ")";
        return out.str();
    }

    const bool readable = access(path.c_str(), R_OK) == 0;
    std::ostringstream out;
    out << "size=" << st.st_size
        << ", mode=" << std::oct << (st.st_mode & 0777) << std::dec
        << ", readable=" << (readable ? "true" : "false");
    if (!readable) {
        out << ", access_error=" << std::strerror(errno) << " (" << errno << ")";
    }
    return out.str();
}

const char * backend_dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
        default:                             return "UNKNOWN";
    }
}

std::string describe_backend_devices() {
    std::ostringstream out;
    out << "[";
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (i > 0) {
            out << ",";
        }
        out << "{"
            << "\"name\":\"" << json_escape(ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "") << "\","
            << "\"description\":\"" << json_escape(ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "") << "\","
            << "\"type\":\"" << backend_dev_type_name(ggml_backend_dev_type(dev)) << "\""
            << "}";
    }
    out << "]";
    return out.str();
}

bool has_gpu_backend_device() {
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

int default_thread_count() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 2;
    }
    return std::max(1, std::min(4, (int) hw));
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int n = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
    if (n < 0) {
        buffer.resize((size_t) -n);
        n = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
    }
    if (n < 0) {
        return {};
    }
    return std::string(buffer.data(), (size_t) n);
}

} // namespace

void init_llama_backend() {
    std::call_once(g_backend_once, [] {
        llama_log_set(android_llama_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

ComputeBackend::ComputeBackend() {
    init_llama_backend();
}

ComputeBackend::~ComputeBackend() {
    release();
}

std::string ComputeBackend::build_prompt(const std::string & user_prompt, bool enable_thinking) const {
    std::string prompt = "<|im_start|>system\n"
                         "You are a helpful local assistant. Think before answering when useful.\n"
                         "<|im_end|>\n"
                         "<|im_start|>user\n" + user_prompt;
    if (enable_thinking) {
        prompt += "\n/think";
    } else {
        prompt += "\n/no_think";
    }
    prompt += "\n<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::string ComputeBackend::load_model(const std::string & model_path) {
    init_llama_backend();
    if (model_path.empty()) {
        return "model path is empty";
    }

    const std::string file_info = describe_file(model_path);
    if (access(model_path.c_str(), R_OK) != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "model file is not readable: " + model_path + " (" + file_info + ")";
        return last_error_;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    available_devices_ = describe_backend_devices();
    cancel_requested_.store(true);
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    const bool try_gpu = llama_supports_gpu_offload() && has_gpu_backend_device();
    model_params.n_gpu_layers = try_gpu ? -1 : 0;
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    active_backend_ = try_gpu ? "llama.cpp Vulkan" : "llama.cpp CPU";
    if (model_ == nullptr && try_gpu) {
        __android_log_write(ANDROID_LOG_WARN, "OSH26Llama", "Vulkan model load failed; retrying CPU backend");
        model_params.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        active_backend_ = "llama.cpp CPU fallback";
    }
    if (model_ == nullptr) {
        model_path_.clear();
        last_error_ = "failed to load model: " + model_path + " (" + file_info + ")";
        return last_error_;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = kDefaultContextSize;
    ctx_params.n_batch = kDefaultBatchSize;
    ctx_params.n_ubatch = kDefaultBatchSize;
    ctx_params.n_seq_max = kDefaultMaxSeq;
    ctx_params.n_threads = default_thread_count();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_params.offload_kqv = try_gpu && active_backend_ == "llama.cpp Vulkan";
    ctx_params.op_offload = try_gpu && active_backend_ == "llama.cpp Vulkan";
    ctx_params.no_perf = false;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr && active_backend_ == "llama.cpp Vulkan") {
        llama_model_free(model_);
        model_params.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        active_backend_ = "llama.cpp CPU fallback";
        ctx_params.offload_kqv = false;
        ctx_params.op_offload = false;
        if (model_ != nullptr) {
            ctx_ = llama_init_from_model(model_, ctx_params);
        }
    }
    if (ctx_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        model_path_.clear();
        last_error_ = "failed to create llama_context";
        return last_error_;
    }

    model_path_ = model_path;
    last_error_.clear();
    cancel_requested_.store(false);
    return "model loaded: " + model_path + " [" + active_backend_ + "] (" + file_info + ")";
}

GenerateResult ComputeBackend::generate(const std::string & user_prompt, const GenerateOptions & options, const TokenCallback & on_token) {
    GenerateResult result;
    std::lock_guard<std::mutex> lock(mutex_);
    if (model_ == nullptr || ctx_ == nullptr) {
        result.error = "model is not loaded";
        last_error_ = result.error;
        return result;
    }
    if (running_) {
        result.error = "generation is already running";
        last_error_ = result.error;
        return result;
    }

    running_ = true;
    cancel_requested_.store(false);
    last_error_.clear();

    const auto start = std::chrono::steady_clock::now();
    const std::string prompt = build_prompt(user_prompt, options.enable_thinking);
    const llama_vocab * vocab = llama_model_get_vocab(model_);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0 || n_prompt >= kDefaultContextSize) {
        result.error = n_prompt >= kDefaultContextSize ? "prompt is too long for current context" : "failed to tokenize prompt";
        last_error_ = result.error;
        running_ = false;
        return result;
    }

    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_tokens.data(), n_prompt, true, true) < 0) {
        result.error = "failed to tokenize prompt";
        last_error_ = result.error;
        running_ = false;
        return result;
    }

    llama_memory_clear(llama_get_memory(ctx_), false);

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    int n_pos = 0;
    bool first_token = true;
    const int max_tokens = std::max(1, std::min(options.max_tokens, kDefaultContextSize - n_prompt));
    bool hit_eog = false;
    bool hit_limit = true;

    for (; n_pos + batch.n_tokens < n_prompt + max_tokens;) {
        if (cancel_requested_.load()) {
            result.cancelled = true;
            hit_limit = false;
            break;
        }

        const int decode_status = llama_decode(ctx_, batch);
        if (decode_status != 0) {
            result.error = "llama_decode failed: " + std::to_string(decode_status);
            hit_limit = false;
            break;
        }

        n_pos += batch.n_tokens;
        llama_token token = llama_sampler_sample(sampler, ctx_, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            hit_eog = true;
            hit_limit = false;
            break;
        }

        if (first_token) {
            const auto now = std::chrono::steady_clock::now();
            result.ttft_ms = std::chrono::duration<double, std::milli>(now - start).count();
            first_token = false;
        }

        std::string piece = token_to_piece(vocab, token);
        result.text += piece;
        if (on_token) {
            on_token(piece);
        }
        batch = llama_batch_get_one(&token, 1);
        result.decoded_tokens += 1;
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::max(0.001, std::chrono::duration<double>(end - start).count());
    result.tokens_per_second = result.decoded_tokens / elapsed_s;
    result.ok = result.error.empty();
    if (result.cancelled) {
        result.finish_reason = "cancelled";
    } else if (!result.error.empty()) {
        result.finish_reason = "error";
    } else if (hit_limit && !hit_eog) {
        result.finish_reason = "length";
    } else {
        result.finish_reason = "stop";
    }

    llama_sampler_free(sampler);
    llama_memory_clear(llama_get_memory(ctx_), false);

    last_ttft_ms_ = result.ttft_ms;
    last_decoded_tokens_ = result.decoded_tokens;
    last_tokens_per_second_ = result.tokens_per_second;
    last_finish_reason_ = result.finish_reason;
    last_error_ = result.error;
    running_ = false;
    return result;
}

void ComputeBackend::cancel() {
    cancel_requested_.store(true);
}

void ComputeBackend::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_requested_.store(true);
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    model_path_.clear();
    running_ = false;
}

std::string ComputeBackend::stats_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "{\n"
        << "  \"backend\": \"" << json_escape(active_backend_) << "\",\n"
        << "  \"gpu_offload_supported\": " << (llama_supports_gpu_offload() ? "true" : "false") << ",\n"
        << "  \"devices\": " << (available_devices_.empty() ? describe_backend_devices() : available_devices_) << ",\n"
        << "  \"api_port\": 8000,\n"
        << "  \"scheduler\": \"lite\",\n"
        << "  \"max_concurrent_requests\": 1,\n"
        << "  \"model_loaded\": " << (model_ ? "true" : "false") << ",\n"
        << "  \"model_path\": \"" << json_escape(model_path_) << "\",\n"
        << "  \"running\": " << (running_ ? "true" : "false") << ",\n"
        << "  \"last_decoded_tokens\": " << last_decoded_tokens_ << ",\n"
        << "  \"last_ttft_ms\": " << last_ttft_ms_ << ",\n"
        << "  \"last_tokens_per_second\": " << last_tokens_per_second_ << ",\n"
        << "  \"last_finish_reason\": \"" << json_escape(last_finish_reason_) << "\",\n"
        << "  \"last_error\": \"" << json_escape(last_error_) << "\"\n"
        << "}";
    return out.str();
}

std::string SchedulerLite::load_model(const std::string & model_path) {
    return backend_.load_model(model_path);
}

GenerateResult SchedulerLite::generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token) {
    return backend_.generate(prompt, options, on_token);
}

void SchedulerLite::cancel() {
    backend_.cancel();
}

void SchedulerLite::release() {
    backend_.release();
}

std::string SchedulerLite::stats_json() const {
    return backend_.stats_json();
}

SchedulerLite & engine() {
    static SchedulerLite instance;
    return instance;
}

} // namespace osh26
