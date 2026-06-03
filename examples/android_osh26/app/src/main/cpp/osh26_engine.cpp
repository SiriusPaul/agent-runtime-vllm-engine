#include "osh26_engine.h"
#include "osh26_vk_gpu.h"

#include <android/log.h>
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace osh26 {
namespace {

constexpr int kDefaultContextSize = 1024;
constexpr int kDefaultBatchSize = 512;
constexpr int kDefaultMaxSeq = 4;
constexpr int kShortPrefillTokenLimit = 64;
constexpr int kMediumPrefillTokenLimit = 256;
constexpr int kShortPrefillChunkSize = 64;
constexpr int kLongPrefillChunkSize = 128;
constexpr int kGpuTopkCandidateCount = 256;
constexpr int kGpuLowIdStreakFallback = 3;

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

std::string describe_osh26_vk_stats() {
    osh26_vk_stats stats {};
    if (osh26_vk_get_stats(&stats) != 0) {
        return "{}";
    }

    std::ostringstream out;
    out << "{"
        << "\"ready\":" << (stats.ready ? "true" : "false") << ","
        << "\"registered\":" << (stats.registered ? "true" : "false") << ","
        << "\"graph_compute_calls\":" << stats.graph_compute_calls << ","
        << "\"mul_mat_dispatches\":" << stats.mul_mat_dispatches << ","
        << "\"rms_norm_dispatches\":" << stats.rms_norm_dispatches << ","
        << "\"buffers_allocated\":" << stats.buffers_allocated << ","
        << "\"buffers_freed\":" << stats.buffers_freed << ","
        << "\"current_buffer_bytes\":" << stats.current_buffer_bytes << ","
        << "\"peak_buffer_bytes\":" << stats.peak_buffer_bytes << ","
        << "\"expanded_tensor_uploads\":" << stats.expanded_tensor_uploads << ","
        << "\"expanded_upload_bytes\":" << stats.expanded_upload_bytes << ","
        << "\"f16_uploads\":" << stats.f16_uploads << ","
        << "\"bf16_uploads\":" << stats.bf16_uploads << ","
        << "\"q4_k_uploads\":" << stats.q4_k_uploads << ","
        << "\"q6_k_uploads\":" << stats.q6_k_uploads << ","
        << "\"mnn_attention_enabled\":" << (stats.mnn_attention_enabled ? "true" : "false") << ","
        << "\"debug_correctness\":" << (stats.debug_correctness ? "true" : "false") << ","
        << "\"mnn_kv_layout\":\"cacheKey[kvHeadNum,headDim/4,maxLen].vec4 cacheValue[kvHeadNum,maxLen,headDim/4].vec4\","
        << "\"last_attention_max_abs_err\":" << stats.last_attention_max_abs_err << ","
        << "\"attention_fallback_layers\":" << stats.attention_fallback_layers << ","
        << "\"last_forward_submit_count\":" << stats.last_forward_submit_count << ","
        << "\"last_forward_layers_ms\":" << stats.last_forward_layers_ms << ","
        << "\"last_forward_attention_ms\":" << stats.last_forward_attention_ms << ","
        << "\"last_forward_kv_update_ms\":" << stats.last_forward_kv_update_ms << ","
        << "\"last_forward_lm_head_ms\":" << stats.last_forward_lm_head_ms << ","
        << "\"last_prefill_ms\":" << stats.last_prefill_ms << ","
        << "\"last_decode_ms\":" << stats.last_decode_ms << ","
        << "\"last_lm_head_ms\":" << stats.last_lm_head_ms << ","
        << "\"last_token_tps\":" << stats.last_token_tps << ","
        << "\"gpu_lm_head_enabled\":" << (stats.gpu_lm_head_enabled ? "true" : "false") << ","
        << "\"last_lm_head_max_abs_err\":" << stats.last_lm_head_max_abs_err << ","
        << "\"last_lm_head_ref_top5\":[" << stats.last_lm_head_ref_top5[0] << ","
        << stats.last_lm_head_ref_top5[1] << ","
        << stats.last_lm_head_ref_top5[2] << ","
        << stats.last_lm_head_ref_top5[3] << ","
        << stats.last_lm_head_ref_top5[4] << "],"
        << "\"last_logits_top5\":["
        << "{\"id\":" << stats.last_logits_top5[0] << ",\"logit\":" << stats.last_logits_top5_values[0] << "},"
        << "{\"id\":" << stats.last_logits_top5[1] << ",\"logit\":" << stats.last_logits_top5_values[1] << "},"
        << "{\"id\":" << stats.last_logits_top5[2] << ",\"logit\":" << stats.last_logits_top5_values[2] << "},"
        << "{\"id\":" << stats.last_logits_top5[3] << ",\"logit\":" << stats.last_logits_top5_values[3] << "},"
        << "{\"id\":" << stats.last_logits_top5[4] << ",\"logit\":" << stats.last_logits_top5_values[4] << "}]"
        << "}";
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
        return 4;
    }
    return std::max(1, std::min(8, (int) hw));
}

std::string token_to_text(const llama_vocab * vocab, llama_token token) {
    int32_t n = llama_detokenize(vocab, &token, 1, nullptr, 0, true, false);
    if (n == 0) {
        return {};
    }
    if (n < 0) {
        n = -n;
    }
    std::string out(static_cast<size_t>(n), '\0');
    const int32_t written = llama_detokenize(vocab, &token, 1, out.data(), n, true, false);
    if (written <= 0) {
        return {};
    }
    out.resize(static_cast<size_t>(written));
    return out;
}

int choose_prefill_chunk_size(int remaining_tokens) {
    if (remaining_tokens <= kShortPrefillTokenLimit) {
        return remaining_tokens;
    }
    if (remaining_tokens <= kMediumPrefillTokenLimit) {
        return kShortPrefillChunkSize;
    }
    return kLongPrefillChunkSize;
}

bool gpu_logits_look_bad(
        const osh26_vk_candidate * candidates,
        int candidate_count,
        int * low_id_streak,
        std::string * reason) {
    if (candidates == nullptr || candidate_count <= 0) {
        if (reason != nullptr) {
            *reason = "gpu logits unavailable";
        }
        return true;
    }
    const int top_count = std::min(candidate_count, 5);
    for (int i = 0; i < top_count; ++i) {
        if (!std::isfinite(candidates[i].logit)) {
            if (reason != nullptr) {
                *reason = "gpu logits contained non-finite values";
            }
            return true;
        }
    }
    if (low_id_streak != nullptr) {
        if (candidates[0].token >= 0 && candidates[0].token <= 20) {
            *low_id_streak += 1;
        } else {
            *low_id_streak = 0;
        }
        if (*low_id_streak >= kGpuLowIdStreakFallback) {
            if (reason != nullptr) {
                *reason = "gpu top token stayed in low-id range";
            }
            return true;
        }
    }
    return false;
}

} // namespace

void init_llama_backend() {
    std::call_once(g_backend_once, [] {
        llama_log_set(android_llama_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        osh26_vk_gpu_init();
    });
}

ComputeBackend::ComputeBackend() {
    init_llama_backend();
}

ComputeBackend::~ComputeBackend() {
    release();
}

std::string ComputeBackend::build_prompt(const std::string & user_prompt, bool enable_thinking) const {
    if (model_ != nullptr) {
        const char * tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl != nullptr && tmpl[0] != '\0') {
            std::string system_msg = "You are a helpful local assistant.";
            std::string user_msg = user_prompt;
            if (enable_thinking) {
                user_msg += "\n/think";
            } else {
                user_msg += "\n/no_think";
            }
            std::array<llama_chat_message, 2> messages = {
                llama_chat_message{"system", system_msg.c_str()},
                llama_chat_message{"user", user_msg.c_str()},
            };
            const int32_t needed = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
            if (needed > 0) {
                std::string buffer(static_cast<size_t>(needed), '\0');
                const int32_t written = llama_chat_apply_template(
                    tmpl,
                    messages.data(),
                    messages.size(),
                    true,
                    buffer.data(),
                    (int32_t) buffer.size());
                if (written > 0) {
                    buffer.resize(static_cast<size_t>(written));
                    return buffer;
                }
            }
        }
    }

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

std::string ComputeBackend::build_prompt_prefix() const {
    if (model_ != nullptr) {
        const char * tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl != nullptr && tmpl[0] != '\0') {
            const std::string marker = "__OSH26_PREFIX_SPLIT_MARKER__";
            std::array<llama_chat_message, 2> messages = {
                llama_chat_message{"system", "You are a helpful local assistant."},
                llama_chat_message{"user", marker.c_str()},
            };
            const int32_t needed = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
            if (needed > 0) {
                std::string buffer(static_cast<size_t>(needed), '\0');
                const int32_t written = llama_chat_apply_template(
                    tmpl,
                    messages.data(),
                    messages.size(),
                    true,
                    buffer.data(),
                    (int32_t) buffer.size());
                if (written > 0) {
                    buffer.resize(static_cast<size_t>(written));
                    const size_t marker_pos = buffer.find(marker);
                    if (marker_pos != std::string::npos) {
                        return buffer.substr(0, marker_pos);
                    }
                }
            }
        }
    }

    return "<|im_start|>system\n"
           "You are a helpful local assistant. Think before answering when useful.\n"
           "<|im_end|>\n"
           "<|im_start|>user\n";
}

bool ComputeBackend::prompt_has_cached_prefix(const std::vector<llama_token> & prompt_tokens) const {
    if (!prefix_cache_valid_ || prefix_tokens_.empty() || prompt_tokens.size() < prefix_tokens_.size()) {
        return false;
    }
    return std::equal(prefix_tokens_.begin(), prefix_tokens_.end(), prompt_tokens.begin());
}

void ComputeBackend::reset_cache_locked(bool clear_prefix_state) {
    if (ctx_ != nullptr) {
        llama_memory_clear(llama_get_memory(ctx_), false);
    }
    if (clear_prefix_state) {
        prefix_cache_valid_ = false;
        prefix_cached_pos_ = 0;
        prefix_tokens_.clear();
        active_prompt_tokens_.clear();
        boot_prefix_tokens_.clear();
        prefix_cache_entry_count_ = 0;
        prefix_cache_token_count_ = 0;
        prefix_cache_evictions_ = 0;
        prefix_cache_hits_ = 0;
        prefix_cache_misses_ = 0;
        prefix_cache_reuse_tokens_ = 0;
        prefix_cache_block_reuse_ = 0;
        prefix_cache_root_ = PrefixCacheNode{};
        prefix_cache_lru_.clear();
        last_prefix_cache_hit_ = false;
        last_prefix_tokens_ = 0;
        last_reusable_prefix_tokens_ = 0;
        last_cached_prefix_entries_ = 0;
    }
}

bool ComputeBackend::warm_prefix_cache_locked() {
    if (model_ == nullptr || ctx_ == nullptr || requested_backend_ != "vulkan" || debug_correctness_ || !osh26_vk_gpu_ready()) {
        return false;
    }
    if (prefix_cache_valid_) {
        return true;
    }

    const std::string prefix = build_prompt_prefix();
    const llama_vocab * vocab = llama_model_get_vocab(model_);
    const int n_prefix = -llama_tokenize(vocab, prefix.c_str(), (int32_t) prefix.size(), nullptr, 0, true, true);
    if (n_prefix <= 0 || n_prefix >= kDefaultContextSize) {
        return false;
    }

    std::vector<llama_token> tokens((size_t) n_prefix);
    if (llama_tokenize(vocab, prefix.c_str(), (int32_t) prefix.size(), tokens.data(), n_prefix, true, true) < 0) {
        return false;
    }

    if (osh26_vk_gpu_forward_ex((int *) tokens.data(), n_prefix, 0, OSH26_FORWARD_PREFILL_ONLY) != 0) {
        __android_log_write(ANDROID_LOG_WARN, "OSH26GPU", "prefix cache warmup failed");
        return false;
    }

    prefix_tokens_ = std::move(tokens);
    prefix_cached_pos_ = n_prefix;
    prefix_cache_valid_ = true;
    prefix_cache_entry_count_ = std::max<size_t>(prefix_cache_entry_count_, 1);
    prefix_cache_token_count_ = std::max<size_t>(prefix_cache_token_count_, (size_t) n_prefix);
    last_cached_prefix_entries_ = prefix_cache_entry_count_;
    __android_log_print(ANDROID_LOG_INFO, "OSH26GPU", "prefix cache warmed tokens=%d", n_prefix);
    return true;
}

size_t ComputeBackend::common_prefix_length(const std::vector<llama_token> & lhs, const std::vector<llama_token> & rhs) const {
    const size_t limit = std::min(lhs.size(), rhs.size());
    size_t n = 0;
    while (n < limit && lhs[n] == rhs[n]) {
        ++n;
    }
    return n;
}

std::shared_ptr<ComputeBackend::GenerationRequest> ComputeBackend::enqueue_request(
        const std::string & prompt,
        const GenerateOptions & options,
        const TokenCallback & on_token) {
    auto request = std::make_shared<GenerationRequest>();
    request->prompt = prompt;
    request->options = options;
    request->on_token = on_token;
    request->submitted_at = std::chrono::steady_clock::now();
    request->reusable_prefix_tokens = 0;
    return request;
}

std::shared_ptr<ComputeBackend::GenerationRequest> ComputeBackend::pick_next_request_locked() {
    if (request_queue_.empty()) {
        return nullptr;
    }

    std::shared_ptr<GenerationRequest> best;
    size_t best_score = 0;
    for (const auto & request : request_queue_) {
        const size_t score = request->reusable_prefix_tokens;
        if (!best || score > best_score || (score == best_score && request->queue_position < best->queue_position)) {
            best = request;
            best_score = score;
        }
    }

    return best;
}

void ComputeBackend::fail_queued_requests_locked(const std::string & error_message, bool cancelled) {
    for (const auto & request : request_queue_) {
        if (request == nullptr) {
            continue;
        }
        std::lock_guard<std::mutex> request_lock(request->mutex);
        request->result.error = error_message;
        request->result.cancelled = cancelled;
        request->result.finish_reason = cancelled ? "cancelled" : "error";
        request->result.ok = false;
        request->completed = true;
        request->cv.notify_all();
        if (cancelled) {
            total_cancelled_requests_ += 1;
        }
    }
    request_queue_.clear();
}

void ComputeBackend::ensure_worker_started_locked() {}

void ComputeBackend::worker_loop() {}

void ComputeBackend::complete_request(const std::shared_ptr<GenerationRequest> & request) {
    if (request == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(request->mutex);
    request->completed = true;
    request->cv.notify_all();
}

void ComputeBackend::insert_prefix_cache_locked(const std::vector<llama_token> & prompt_tokens) {
    const size_t prefix_len = std::min(prompt_tokens.size(), max_prefix_cache_tokens_);
    if (prefix_len == 0) {
        return;
    }
    PrefixCacheNode * node = ensure_prefix_node_locked(std::vector<llama_token>(prompt_tokens.begin(), prompt_tokens.begin() + prefix_len));
    if (node == nullptr) {
        return;
    }
    if (node->terminal) {
        node->request_count += 1;
        node->hit_count += 1;
        touch_prefix_node_locked(node);
    } else {
        node->terminal = true;
        node->token_count = prefix_len;
        node->request_count = 1;
        node->hit_count = 0;
        prefix_cache_lru_.push_front(node);
        prefix_cache_entry_count_ += 1;
        prefix_cache_token_count_ += prefix_len;
    }
    prefix_cache_valid_ = true;
    prefix_cached_pos_ = static_cast<int>(prefix_len);
    prefix_tokens_.assign(prompt_tokens.begin(), prompt_tokens.begin() + prefix_len);
    last_cached_prefix_entries_ = prefix_cache_entry_count_;
    evict_prefix_cache_locked();
}

ComputeBackend::PrefixCacheNode * ComputeBackend::ensure_prefix_node_locked(const std::vector<llama_token> & tokens) {
    PrefixCacheNode * node = &prefix_cache_root_;
    for (const llama_token token : tokens) {
        auto & child = node->children[token];
        if (!child) {
            child = std::make_unique<PrefixCacheNode>();
            child->parent = node;
            child->token = token;
        }
        node = child.get();
    }
    return node;
}

ComputeBackend::PrefixCacheNode * ComputeBackend::find_prefix_node_locked(const std::vector<llama_token> & tokens) const {
    const PrefixCacheNode * node = &prefix_cache_root_;
    for (const llama_token token : tokens) {
        auto it = node->children.find(token);
        if (it == node->children.end()) {
            return nullptr;
        }
        node = it->second.get();
    }
    return const_cast<PrefixCacheNode *>(node);
}

void ComputeBackend::touch_prefix_node_locked(PrefixCacheNode * node) {
    if (node == nullptr) {
        return;
    }
    prefix_cache_lru_.remove(node);
    prefix_cache_lru_.push_front(node);
    node->last_used_tick += 1;
}

void ComputeBackend::evict_prefix_cache_locked() {
    while (prefix_cache_entry_count_ > max_prefix_cache_entries_ && !prefix_cache_lru_.empty()) {
        PrefixCacheNode * victim = prefix_cache_lru_.back();
        prefix_cache_lru_.pop_back();
        if (victim != nullptr && victim->terminal) {
            victim->terminal = false;
            if (prefix_cache_token_count_ >= victim->token_count) {
                prefix_cache_token_count_ -= victim->token_count;
            } else {
                prefix_cache_token_count_ = 0;
            }
            victim->token_count = 0;
            if (prefix_cache_entry_count_ > 0) {
                prefix_cache_entry_count_ -= 1;
            }
            prefix_cache_evictions_ += 1;
        }
    }
}

size_t ComputeBackend::prefix_cache_entry_count_locked() const {
    return prefix_cache_entry_count_;
}

size_t ComputeBackend::prefix_cache_token_count_locked() const {
    return prefix_cache_token_count_;
}

double ComputeBackend::prefix_cache_fragmentation_locked() const {
    if (prefix_cache_token_count_ == 0) {
        return 0.0;
    }
    const size_t used_tokens = std::min(prefix_tokens_.size(), prefix_cache_token_count_);
    const double total_tokens = (double) std::max<size_t>(1, prefix_cache_token_count_);
    return 1.0 - ((double) used_tokens / total_tokens);
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

    release();

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
    osh26_vk_gpu_free();
    active_backend_ = "llama.cpp CPU";
    reset_cache_locked(true);

    // CPU-only for llama.cpp (n_gpu_layers=0), weights go to GPU pool separately
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model_ == nullptr) {
        model_path_.clear();
        last_error_ = "failed to load model: " + model_path + " (" + file_info + ")";
        return last_error_;
    }
    if (requested_backend_ == "vulkan") {
        // Load weights into the handwritten Vulkan runtime only when explicitly requested.
        if (osh26_vk_gpu_load_model(model_path.c_str()) == 0) {
            active_backend_ = "OSH26 GPU Runtime";
            __android_log_write(ANDROID_LOG_INFO, "OSH26Llama", "GPU model loaded");
        } else {
            active_backend_ = "CPU (GPU load failed)";
            __android_log_write(ANDROID_LOG_WARN, "OSH26Llama", "GPU model load failed");
        }
    } else {
        __android_log_write(ANDROID_LOG_INFO, "OSH26Llama", "Skipping GPU model load (CPU backend requested)");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = kDefaultContextSize;
    ctx_params.n_batch = kDefaultBatchSize;
    ctx_params.n_ubatch = kDefaultBatchSize;
    ctx_params.n_seq_max = kDefaultMaxSeq;
    ctx_params.n_threads = default_thread_count();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    // The custom Vulkan backend does not implement the KV update/attention ops yet.
    // Keep KV cache in CPU memory while allowing layer weights and supported matmuls on GPU.
    ctx_params.offload_kqv = false;
    ctx_params.op_offload = false;
    ctx_params.no_perf = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        osh26_vk_gpu_free();
        llama_model_free(model_);
        model_ = nullptr;
        model_path_.clear();
        active_backend_ = "llama.cpp CPU";
        last_error_ = "failed to create llama_context";
        return last_error_;
    }

    model_path_ = model_path;
    last_error_.clear();
    last_prefill_forward_count_ = 0;
    last_prefill_skipped_lm_head_count_ = 0;
    last_prefill_chunk_size_ = 0;
    last_prefill_chunk_count_ = 0;
    last_prefill_logits_chunks_ = 0;
    last_logits_sanity_ok_ = true;
    last_logits_low_id_streak_ = 0;
    last_logits_sanity_reason_.clear();
    cancel_requested_.store(false);
    warm_prefix_cache_locked();
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        shutdown_requested_ = false;
        queue_cv_.notify_all();
    }
    return "model loaded: " + model_path + " [" + active_backend_ + "] (" + file_info + ")";
}

GenerateResult ComputeBackend::generate(const std::string & user_prompt, const GenerateOptions & options, const TokenCallback & on_token) {
    GenerateResult result;
    auto request = std::make_shared<GenerationRequest>();
    request->options = options;
    request->on_token = on_token;
    request->submitted_at = std::chrono::steady_clock::now();

    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    int n_prompt = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (model_ == nullptr || ctx_ == nullptr) {
            result.error = "model is not loaded";
            last_error_ = result.error;
            return result;
        }

        last_error_.clear();
        last_prompt_build_ms_ = 0.0;
        last_tokenize_ms_ = 0.0;
        last_prefix_cache_hit_ = false;
        last_prefix_tokens_ = 0;
        last_user_prefill_tokens_ = 0;
        last_prefill_forward_count_ = 0;
        last_prefill_skipped_lm_head_count_ = 0;
        last_prefill_chunk_size_ = 0;
        last_prefill_chunk_count_ = 0;
        last_prefill_logits_chunks_ = 0;
        last_user_prefill_ms_ = 0.0;
        last_first_decode_ms_ = 0.0;
        last_logits_sanity_ok_ = true;
        last_logits_low_id_streak_ = 0;
        last_logits_sanity_reason_.clear();
    last_logits_sanity_ok_ = true;
    last_logits_low_id_streak_ = 0;
    last_logits_sanity_reason_.clear();

        const auto prompt_build_start = std::chrono::steady_clock::now();
        prompt = build_prompt(user_prompt, options.enable_thinking);
        last_prompt_build_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prompt_build_start).count();

        const auto tokenize_start = std::chrono::steady_clock::now();
        const llama_vocab * vocab = llama_model_get_vocab(model_);
        n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0 || n_prompt >= kDefaultContextSize) {
            result.error = n_prompt >= kDefaultContextSize ? "prompt is too long for current context" : "failed to tokenize prompt";
            last_error_ = result.error;
            return result;
        }

        prompt_tokens.resize((size_t) n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_tokens.data(), n_prompt, true, true) < 0) {
            result.error = "failed to tokenize prompt";
            last_error_ = result.error;
            return result;
        }
        last_tokenize_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tokenize_start).count();

        request->prompt = std::move(prompt);
        request->prompt_tokens = prompt_tokens;
        request->prompt_tokens_total = request->prompt_tokens.size();
        request->reusable_prefix_tokens = common_prefix_length(request->prompt_tokens, prefix_tokens_);
        request->queue_position = next_queue_position_++;

    }

    {
        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        if (request_queue_.size() >= max_pending_requests_) {
            result.error = "request queue is full";
            total_rejected_requests_ += 1;
            return result;
        }
        request_queue_.push_back(request);
        total_submitted_requests_ += 1;
        queue_depth_peak_ = std::max(queue_depth_peak_, request_queue_.size());
        queue_cv_.wait(queue_lock, [&] {
            if (shutdown_requested_) {
                return true;
            }
            const auto next = pick_next_request_locked();
            return active_request_ == nullptr && next == request;
        });
        if (shutdown_requested_) {
            auto it = std::find(request_queue_.begin(), request_queue_.end(), request);
            if (it != request_queue_.end()) {
                request_queue_.erase(it);
            }
            total_cancelled_requests_ += 1;
            result.error = "engine is shutting down";
            result.cancelled = true;
            result.finish_reason = "cancelled";
            return result;
        }
        active_request_ = request;
        running_ = true;
        request->started = true;
        request->started_at = std::chrono::steady_clock::now();
        request->queue_wait_ms = std::chrono::duration<double, std::milli>(request->started_at - request->submitted_at).count();
        last_queue_wait_ms_ = request->queue_wait_ms;
        total_queue_wait_ms_ += request->queue_wait_ms;
        auto it = std::find(request_queue_.begin(), request_queue_.end(), request);
        if (it != request_queue_.end()) {
            request_queue_.erase(it);
        }
    }

    cancel_requested_.store(request->cancelled);
    const auto start = request->submitted_at;
    const llama_vocab * vocab = llama_model_get_vocab(model_);

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));

    const int max_tokens = std::max(1, std::min(options.max_tokens, kDefaultContextSize - n_prompt));
    bool hit_eog = false;
    bool hit_limit = true;

    const bool want_gpu = requested_backend_ == "vulkan";
    bool use_gpu = want_gpu && osh26_vk_gpu_ready();
    if(use_gpu)__android_log_print(ANDROID_LOG_INFO,"OSH26GPU","VOCAB check: hardcoded=151936 llama_vocab=%d",(int)llama_vocab_n_tokens(vocab));
    int n_pos = 0;
    bool first_token = true;

    if (use_gpu) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            warm_prefix_cache_locked();
            if (prompt_has_cached_prefix(prompt_tokens)) {
                last_prefix_cache_hit_ = true;
                last_prefix_tokens_ = prefix_cached_pos_;
                n_pos = prefix_cached_pos_;
            } else {
                prefix_cache_valid_ = false;
                prefix_cached_pos_ = 0;
                prefix_tokens_.clear();
            }
        }

        const int prefill_start = n_pos;
        const int user_prefill_tokens = n_prompt - prefill_start;
        last_user_prefill_tokens_ = std::max(0, user_prefill_tokens);
        last_prefill_chunk_size_ = 0;
        last_prefill_chunk_count_ = 0;
        last_prefill_logits_chunks_ = 0;
        last_logits_sanity_ok_ = true;
        last_logits_low_id_streak_ = 0;
        last_logits_sanity_reason_.clear();
        if (user_prefill_tokens > 0) {
            const auto prefill_time_start = std::chrono::steady_clock::now();
            const llama_token * prefill_tokens = prompt_tokens.data() + prefill_start;
            const int chunk_size = choose_prefill_chunk_size(user_prefill_tokens);
            last_prefill_chunk_size_ = chunk_size;
            for (int offset = 0; offset < user_prefill_tokens; offset += chunk_size) {
                if (cancel_requested_.load()) { result.cancelled = true; hit_limit = false; break; }
                const int chunk_tokens = std::min(chunk_size, user_prefill_tokens - offset);
                const bool needs_logits = (offset + chunk_tokens == user_prefill_tokens);
                const uint32_t forward_flags = needs_logits ? OSH26_FORWARD_NEED_LOGITS : OSH26_FORWARD_PREFILL_ONLY;
                ++last_prefill_forward_count_;
                ++last_prefill_chunk_count_;
                if (needs_logits) {
                    ++last_prefill_logits_chunks_;
                } else {
                    ++last_prefill_skipped_lm_head_count_;
                }
                if (osh26_vk_gpu_forward_ex((int *) (prefill_tokens + offset), chunk_tokens, n_pos, forward_flags) != 0) {
                    if (result.decoded_tokens == 0) {
                        use_gpu = false;
                        result.error.clear();
                        goto cpu_path;
                    }
                    result.error = needs_logits ? "GPU prefill failed" : "GPU prefill chunk failed";
                    hit_limit = false;
                    break;
                }
                n_pos += chunk_tokens;
            }
            last_user_prefill_ms_ = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefill_time_start).count();
        } else {
            last_user_prefill_ms_ = 0.0;
            const bool can_refresh_tail = n_prompt > 0
                && last_prefix_cache_hit_
                && prefix_cached_pos_ == n_prompt
                && n_pos > 0
                && !prefix_tokens_.empty()
                && prompt_tokens.back() == prefix_tokens_.back();
            if (can_refresh_tail) {
                const auto prefill_time_start = std::chrono::steady_clock::now();
                int refresh_token = (int) prompt_tokens[n_prompt - 1];
                ++last_prefill_forward_count_;
                if (osh26_vk_gpu_forward_ex(&refresh_token, 1, std::max(0, n_pos - 1), OSH26_FORWARD_NEED_LOGITS) != 0) {
                    if (result.decoded_tokens == 0) {
                        use_gpu = false;
                        result.error.clear();
                        goto cpu_path;
                    }
                    result.error = "GPU cache refresh failed";
                    hit_limit = false;
                }
                last_user_prefill_ms_ = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prefill_time_start).count();
            }
        }

        for (; result.error.empty() && !result.cancelled && result.decoded_tokens < max_tokens;) {
            if (cancel_requested_.load()) { result.cancelled = true; hit_limit = false; break; }

            const auto first_decode_start = first_token ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

            osh26_vk_candidate gpu_candidates[kGpuTopkCandidateCount];
            const int candidate_count = osh26_vk_gpu_collect_topk(gpu_candidates, kGpuTopkCandidateCount);
            if (candidate_count <= 0) {
                if (result.decoded_tokens == 0) {
                    use_gpu = false;
                    goto cpu_path;
                }
                result.error = "GPU logits candidate collection failed";
                hit_limit = false;
                break;
            }

            if (gpu_logits_look_bad(gpu_candidates, candidate_count, &last_logits_low_id_streak_, &last_logits_sanity_reason_)) {
                last_logits_sanity_ok_ = false;
                __android_log_print(ANDROID_LOG_WARN, "OSH26GPU", "GPU logits sanity fallback: %s", last_logits_sanity_reason_.c_str());
                if (result.decoded_tokens == 0) {
                    use_gpu = false;
                    result.error.clear();
                    goto cpu_path;
                }
                result.error = "GPU logits sanity check failed: " + last_logits_sanity_reason_;
                hit_limit = false;
                break;
            }
            last_logits_sanity_ok_ = true;
            last_logits_sanity_reason_.clear();

            static std::vector<llama_token_data> s_candidates;
            if ((int)s_candidates.size() < candidate_count) {
                s_candidates.resize((size_t) candidate_count);
            }
            for (int i = 0; i < candidate_count; ++i) {
                s_candidates[i] = { (llama_token) gpu_candidates[i].token, gpu_candidates[i].logit, 0.0f };
            }
            llama_token_data_array cur_p = {
                s_candidates.data(), (size_t) candidate_count, -1, false };
            llama_sampler_apply(sampler, &cur_p);
            if (cur_p.selected < 0 || (size_t) cur_p.selected >= cur_p.size) {
                result.error = "sampler returned invalid index";
                hit_limit = false;
                break;
            }
            llama_token token = cur_p.data[cur_p.selected].id;

            if (llama_vocab_is_eog(vocab, token)) { hit_eog = true; hit_limit = false; break; }
            llama_sampler_accept(sampler, token);

            if (first_token) {
                last_first_decode_ms_ = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - first_decode_start).count();
                result.ttft_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                first_token = false;
            }

            std::string piece = token_to_text(vocab, token);
            if (!result.token_ids.empty()) result.token_ids += ",";
            result.token_ids += std::to_string(token);
            if (!piece.empty()) { result.text += piece; if (on_token) on_token(piece); }
            result.decoded_tokens += 1;
            if (result.decoded_tokens >= max_tokens) {
                break;
            }

            const auto decode_start = std::chrono::steady_clock::now();
            if (osh26_vk_gpu_forward_ex((int *) &token, 1, n_pos, OSH26_FORWARD_NEED_LOGITS) != 0) {
                result.error = "GPU decode failed";
                hit_limit = false;
                break;
            }
            n_pos += 1;
            if (first_token) {
                last_first_decode_ms_ = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - decode_start).count();
            }
        }
    }
cpu_path:
    if (!use_gpu) {
        llama_memory_clear(llama_get_memory(ctx_), false);
        n_pos = 0;
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
        last_user_prefill_tokens_ = n_prompt;
        last_prefill_forward_count_ = 1;
        last_prefill_skipped_lm_head_count_ = 0;
        last_prefill_chunk_size_ = n_prompt;
        last_prefill_chunk_count_ = n_prompt > 0 ? 1 : 0;
        last_prefill_logits_chunks_ = n_prompt > 0 ? 1 : 0;
        last_logits_sanity_ok_ = true;
        last_logits_low_id_streak_ = 0;
        last_logits_sanity_reason_.clear();
        for (; n_pos + batch.n_tokens < n_prompt + max_tokens;) {
            if (cancel_requested_.load()) { result.cancelled = true; hit_limit = false; break; }

            // CPU path via llama_decode
            const auto decode_start = std::chrono::steady_clock::now();
            const int decode_status = llama_decode(ctx_, batch);
            if (decode_status != 0) {
                result.error = "llama_decode failed: " + std::to_string(decode_status);
                hit_limit = false; break;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            if (first_token) {
                last_user_prefill_ms_ = decode_ms;
            } else if (last_first_decode_ms_ == 0.0) {
                last_first_decode_ms_ = decode_ms;
            }

            n_pos += batch.n_tokens;
            llama_token token = llama_sampler_sample(sampler, ctx_, -1);
            if (llama_vocab_is_eog(vocab, token)) { hit_eog = true; hit_limit = false; break; }

            if (first_token) {
                result.ttft_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                first_token = false;
            }

            std::string piece = token_to_text(vocab, token);
            if (!result.token_ids.empty()) result.token_ids += ",";
            result.token_ids += std::to_string(token);
            if (!piece.empty()) { result.text += piece; if (on_token) on_token(piece); }
            batch = llama_batch_get_one(&token, 1);
            result.decoded_tokens += 1;
        }
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

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        active_request_.reset();
        running_ = false;
        if (result.cancelled) {
            total_cancelled_requests_ += 1;
        } else {
            total_completed_requests_ += 1;
        }
        queue_cv_.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_ttft_ms_ = result.ttft_ms;
        last_decoded_tokens_ = result.decoded_tokens;
        last_tokens_per_second_ = result.tokens_per_second;
        last_finish_reason_ = result.finish_reason;
        last_error_ = result.error;
        last_token_ids_ = result.token_ids;
        if (result.ok && !result.cancelled && result.error.empty() && use_gpu) {
            insert_prefix_cache_locked(prompt_tokens);
            last_prefix_tokens_ = prefix_cached_pos_;
            last_prefix_cache_hit_ = last_prefix_tokens_ > 0 && request->reusable_prefix_tokens > 0;
            if (request->reusable_prefix_tokens > 0) {
                prefix_cache_hits_ += 1;
                prefix_cache_reuse_tokens_ += request->reusable_prefix_tokens;
                prefix_cache_block_reuse_ += (request->reusable_prefix_tokens + prefix_cache_block_size_ - 1) / std::max<size_t>(1, prefix_cache_block_size_);
            } else {
                prefix_cache_misses_ += 1;
            }
        } else {
            prefix_cache_valid_ = false;
            prefix_cached_pos_ = 0;
            prefix_tokens_.clear();
        }
    }
    return result;
}

void ComputeBackend::configure_backend(const std::string & mode, int n_gpu_layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string old_backend = requested_backend_;
    if (mode == "cpu" || mode == "vulkan") {
        requested_backend_ = mode;
    } else {
        requested_backend_ = "vulkan";
    }
    requested_gpu_layers_ = n_gpu_layers;
    if (old_backend != requested_backend_) {
        reset_cache_locked(true);
    }
}

void ComputeBackend::set_debug_correctness(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    debug_correctness_ = enabled;
    osh26_vk_gpu_set_debug_correctness(enabled);
    reset_cache_locked(true);
}

void ComputeBackend::reset_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    reset_cache_locked(true);
}

void ComputeBackend::cancel() {
    cancel_requested_.store(true);
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    if (active_request_ != nullptr) {
        std::lock_guard<std::mutex> request_lock(active_request_->mutex);
        active_request_->cancelled = true;
    } else if (!request_queue_.empty()) {
        std::lock_guard<std::mutex> request_lock(request_queue_.front()->mutex);
        request_queue_.front()->cancelled = true;
    }
    queue_cv_.notify_all();
}

void ComputeBackend::release() {
    std::shared_ptr<GenerationRequest> active_request;
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        shutdown_requested_ = true;
        cancel_requested_.store(true);
        active_request = active_request_;
        fail_queued_requests_locked("engine released", true);
        queue_cv_.notify_all();
    }
    if (active_request != nullptr) {
        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        queue_cv_.wait(queue_lock, [&] {
            return active_request_ == nullptr;
        });
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ctx_ != nullptr) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_ != nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
        }
        osh26_vk_gpu_free();
        model_path_.clear();
        active_backend_ = "llama.cpp CPU";
        reset_cache_locked(true);
        last_prefill_forward_count_ = 0;
        last_prefill_skipped_lm_head_count_ = 0;
        last_prefill_chunk_size_ = 0;
        last_prefill_chunk_count_ = 0;
        last_prefill_logits_chunks_ = 0;
        last_logits_sanity_ok_ = true;
        last_logits_low_id_streak_ = 0;
        last_logits_sanity_reason_.clear();
        running_ = false;
    }

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        queue_cv_.notify_all();
    }
}

std::string ComputeBackend::stats_json() const {
    size_t queued_requests = 0;
    bool has_active_request = false;
    bool shutdown = false;
    size_t queue_depth_peak = 0;
    size_t total_submitted_requests = 0;
    size_t total_rejected_requests = 0;
    size_t total_cancelled_requests = 0;
    size_t total_completed_requests = 0;
    double total_queue_wait_ms = 0.0;
    double last_queue_wait_ms = 0.0;
    bool running = false;
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        queued_requests = request_queue_.size();
        has_active_request = active_request_ != nullptr;
        shutdown = shutdown_requested_;
        running = running_;
        queue_depth_peak = queue_depth_peak_;
        total_submitted_requests = total_submitted_requests_;
        total_rejected_requests = total_rejected_requests_;
        total_cancelled_requests = total_cancelled_requests_;
        total_completed_requests = total_completed_requests_;
        total_queue_wait_ms = total_queue_wait_ms_;
        last_queue_wait_ms = last_queue_wait_ms_;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const double prefix_hit_ratio = (prefix_cache_hits_ + prefix_cache_misses_) > 0
        ? (double) prefix_cache_hits_ / (double) (prefix_cache_hits_ + prefix_cache_misses_)
        : 0.0;
    const double block_reuse_ratio = prefix_cache_entry_count_ > 0
        ? (double) prefix_cache_reuse_tokens_ / (double) std::max<size_t>(1, prefix_cache_token_count_)
        : 0.0;
    std::ostringstream out;
    out << "{\n"
        << "  \"backend\": \"" << json_escape(active_backend_) << "\",\n"
        << "  \"requested_backend\": \"" << json_escape(requested_backend_) << "\",\n"
        << "  \"requested_gpu_layers\": " << requested_gpu_layers_ << ",\n"
        << "  \"debug_correctness\": " << (debug_correctness_ ? "true" : "false") << ",\n"
        << "  \"gpu_offload_supported\": " << (llama_supports_gpu_offload() ? "true" : "false") << ",\n"
        << "  \"kv_cache_device\": \"CPU\",\n"
        << "  \"max_context_tokens\": " << kDefaultContextSize << ",\n"
        << "  \"short_prefill_token_limit\": " << kShortPrefillTokenLimit << ",\n"
        << "  \"vulkan\": " << describe_osh26_vk_stats() << ",\n"
        << "  \"devices\": " << (available_devices_.empty() ? describe_backend_devices() : available_devices_) << ",\n"
        << "  \"api_port\": 8000,\n"
        << "  \"scheduler\": \"queued-prefix-lite\",\n"
        << "  \"max_concurrent_requests\": 1,\n"
        << "  \"max_pending_requests\": " << max_pending_requests_ << ",\n"
        << "  \"model_loaded\": " << (model_ ? "true" : "false") << ",\n"
        << "  \"model_path\": \"" << json_escape(model_path_) << "\",\n"
        << "  \"running\": " << (running ? "true" : "false") << ",\n"
        << "  \"shutdown_requested\": " << (shutdown ? "true" : "false") << ",\n"
        << "  \"queue_depth\": " << queued_requests << ",\n"
        << "  \"active_requests\": " << (has_active_request ? 1 : 0) << ",\n"
        << "  \"queue_depth_peak\": " << queue_depth_peak << ",\n"
        << "  \"total_submitted_requests\": " << total_submitted_requests << ",\n"
        << "  \"total_rejected_requests\": " << total_rejected_requests << ",\n"
        << "  \"total_cancelled_requests\": " << total_cancelled_requests << ",\n"
        << "  \"total_completed_requests\": " << total_completed_requests << ",\n"
        << "  \"last_queue_wait_ms\": " << last_queue_wait_ms << ",\n"
        << "  \"avg_queue_wait_ms\": " << (total_submitted_requests > 0 ? total_queue_wait_ms / (double) total_submitted_requests : 0.0) << ",\n"
        << "  \"last_decoded_tokens\": " << last_decoded_tokens_ << ",\n"
        << "  \"last_prompt_build_ms\": " << last_prompt_build_ms_ << ",\n"
        << "  \"last_tokenize_ms\": " << last_tokenize_ms_ << ",\n"
        << "  \"last_prefix_cache_hit\": " << (last_prefix_cache_hit_ ? "true" : "false") << ",\n"
        << "  \"prefix_cache_valid\": " << (prefix_cache_valid_ ? "true" : "false") << ",\n"
        << "  \"prefix_cached_pos\": " << prefix_cached_pos_ << ",\n"
        << "  \"last_prefix_tokens\": " << last_prefix_tokens_ << ",\n"
        << "  \"prefix_cache_entries\": " << prefix_cache_entry_count_ << ",\n"
        << "  \"prefix_cache_tokens\": " << prefix_cache_token_count_ << ",\n"
        << "  \"prefix_cache_evictions\": " << prefix_cache_evictions_ << ",\n"
        << "  \"prefix_cache_hit_ratio\": " << prefix_hit_ratio << ",\n"
        << "  \"prefix_cache_block_reuse_ratio\": " << block_reuse_ratio << ",\n"
        << "  \"prefix_cache_fragmentation\": " << prefix_cache_fragmentation_locked() << ",\n"
        << "  \"last_user_prefill_tokens\": " << last_user_prefill_tokens_ << ",\n"
        << "  \"last_prefill_forward_count\": " << last_prefill_forward_count_ << ",\n"
        << "  \"last_prefill_skipped_lm_head_count\": " << last_prefill_skipped_lm_head_count_ << ",\n"
        << "  \"last_prefill_chunk_size\": " << last_prefill_chunk_size_ << ",\n"
        << "  \"last_prefill_chunk_count\": " << last_prefill_chunk_count_ << ",\n"
        << "  \"last_prefill_logits_chunks\": " << last_prefill_logits_chunks_ << ",\n"
        << "  \"last_user_prefill_ms\": " << last_user_prefill_ms_ << ",\n"
        << "  \"last_first_decode_ms\": " << last_first_decode_ms_ << ",\n"
        << "  \"last_logits_sanity_ok\": " << (last_logits_sanity_ok_ ? "true" : "false") << ",\n"
        << "  \"last_logits_low_id_streak\": " << last_logits_low_id_streak_ << ",\n"
        << "  \"last_logits_sanity_reason\": \"" << json_escape(last_logits_sanity_reason_) << "\",\n"
        << "  \"last_ttft_ms\": " << last_ttft_ms_ << ",\n"
        << "  \"last_tokens_per_second\": " << last_tokens_per_second_ << ",\n"
        << "  \"last_finish_reason\": \"" << json_escape(last_finish_reason_) << "\",\n"
        << "  \"last_error\": \"" << json_escape(last_error_) << "\",\n"
        << "  \"last_token_ids\": \"" << json_escape(last_token_ids_) << "\"\n"
        << "}";
    return out.str();
}

std::string SchedulerLite::load_model(const std::string & model_path) {
    return backend_.load_model(model_path);
}

GenerateResult SchedulerLite::generate(const std::string & prompt, const GenerateOptions & options, const TokenCallback & on_token) {
    return backend_.generate(prompt, options, on_token);
}

void SchedulerLite::configure_backend(const std::string & mode, int n_gpu_layers) {
    backend_.configure_backend(mode, n_gpu_layers);
}

void SchedulerLite::set_debug_correctness(bool enabled) {
    backend_.set_debug_correctness(enabled);
}

void SchedulerLite::reset_cache() {
    backend_.reset_cache();
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
