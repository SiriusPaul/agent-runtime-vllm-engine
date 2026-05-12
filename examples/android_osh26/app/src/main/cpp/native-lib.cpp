#include <jni.h>

#include <android/log.h>
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kDefaultContextSize = 2048;
constexpr int kDefaultMaxTokens = 128;

JavaVM * g_vm = nullptr;
std::once_flag g_backend_once;

struct EngineState {
    std::mutex mutex;
    llama_model * model = nullptr;
    std::string model_path;
    std::thread worker;
    std::atomic_bool cancel_requested{false};
    bool running = false;
    int last_decoded_tokens = 0;
    double last_ttft_ms = 0.0;
    double last_tokens_per_second = 0.0;
    std::string last_error;
};

EngineState g_engine;

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

void ensure_backend() {
    std::call_once(g_backend_once, [] {
        llama_log_set(android_llama_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
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

jstring string_to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
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

int default_thread_count() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 2;
    }
    return std::max(1, std::min(4, (int) hw));
}

std::string stats_json_locked() {
    std::ostringstream out;
    out << "{\n"
        << "  \"backend\": \"llama.cpp CPU\",\n"
        << "  \"model_loaded\": " << (g_engine.model ? "true" : "false") << ",\n"
        << "  \"model_path\": \"" << json_escape(g_engine.model_path) << "\",\n"
        << "  \"running\": " << (g_engine.running ? "true" : "false") << ",\n"
        << "  \"last_decoded_tokens\": " << g_engine.last_decoded_tokens << ",\n"
        << "  \"last_ttft_ms\": " << g_engine.last_ttft_ms << ",\n"
        << "  \"last_tokens_per_second\": " << g_engine.last_tokens_per_second << ",\n"
        << "  \"last_error\": \"" << json_escape(g_engine.last_error) << "\"\n"
        << "}";
    return out.str();
}

void call_callback(jobject callback, const char * method_name, const std::string & value) {
    if (g_vm == nullptr || callback == nullptr) {
        return;
    }

    JNIEnv * env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
            return;
        }
        attached = true;
    }

    jclass callback_class = env->GetObjectClass(callback);
    jmethodID method = env->GetMethodID(callback_class, method_name, "(Ljava/lang/String;)V");
    if (method != nullptr) {
        jstring payload = env->NewStringUTF(value.c_str());
        env->CallVoidMethod(callback, method, payload);
        env->DeleteLocalRef(payload);
    }
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    env->DeleteLocalRef(callback_class);

    if (attached) {
        g_vm->DetachCurrentThread();
    }
}

void finish_generation(double ttft_ms, int decoded_tokens, double tokens_per_second, const std::string & error) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    g_engine.running = false;
    g_engine.last_ttft_ms = ttft_ms;
    g_engine.last_decoded_tokens = decoded_tokens;
    g_engine.last_tokens_per_second = tokens_per_second;
    g_engine.last_error = error;
}

void stop_generation_locked(std::unique_lock<std::mutex> & lock) {
    if (g_engine.worker.joinable()) {
        g_engine.cancel_requested.store(true);
        std::thread worker = std::move(g_engine.worker);
        lock.unlock();
        worker.join();
        lock.lock();
    }
}

void join_finished_worker_locked(std::unique_lock<std::mutex> & lock) {
    if (!g_engine.running && g_engine.worker.joinable()) {
        std::thread worker = std::move(g_engine.worker);
        lock.unlock();
        worker.join();
        lock.lock();
    }
}

void run_generation(jobject callback, llama_model * model, std::string prompt) {
    const auto start = std::chrono::steady_clock::now();
    double ttft_ms = 0.0;
    int decoded_tokens = 0;
    std::string error;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        error = "failed to tokenize prompt";
        call_callback(callback, "onError", error);
        finish_generation(ttft_ms, decoded_tokens, 0.0, error);
        return;
    }

    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_tokens.data(), n_prompt, true, true) < 0) {
        error = "failed to tokenize prompt";
        call_callback(callback, "onError", error);
        finish_generation(ttft_ms, decoded_tokens, 0.0, error);
        return;
    }

    const int n_ctx = std::max(kDefaultContextSize, n_prompt + kDefaultMaxTokens);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) n_ctx;
    ctx_params.n_batch = (uint32_t) std::max(128, n_prompt);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_threads = default_thread_count();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        error = "failed to create llama_context";
        call_callback(callback, "onError", error);
        finish_generation(ttft_ms, decoded_tokens, 0.0, error);
        return;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    int n_pos = 0;
    bool first_token = true;

    for (; n_pos + batch.n_tokens < n_prompt + kDefaultMaxTokens;) {
        if (g_engine.cancel_requested.load()) {
            break;
        }

        if (llama_decode(ctx, batch) != 0) {
            error = "llama_decode failed";
            call_callback(callback, "onError", error);
            break;
        }

        n_pos += batch.n_tokens;
        llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        if (first_token) {
            const auto now = std::chrono::steady_clock::now();
            ttft_ms = std::chrono::duration<double, std::milli>(now - start).count();
            first_token = false;
        }

        std::string piece = token_to_piece(vocab, new_token_id);
        call_callback(callback, "onToken", piece);
        batch = llama_batch_get_one(&new_token_id, 1);
        decoded_tokens += 1;
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::max(0.001, std::chrono::duration<double>(end - start).count());
    const double tokens_per_second = decoded_tokens / elapsed_s;

    std::ostringstream final_stats;
    final_stats << "decoded_tokens=" << decoded_tokens
                << ", ttft_ms=" << ttft_ms
                << ", tokens_per_second=" << tokens_per_second
                << (g_engine.cancel_requested.load() ? ", cancelled=true" : ", cancelled=false");

    if (error.empty()) {
        call_callback(callback, "onComplete", final_stats.str());
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    finish_generation(ttft_ms, decoded_tokens, tokens_per_second, error);
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_MainActivity_nativeLoadModel(JNIEnv * env, jobject, jstring j_model_path) {
    ensure_backend();
    std::string model_path = jstring_to_string(env, j_model_path);
    if (model_path.empty()) {
        return string_to_jstring(env, "model path is empty");
    }

    const std::string file_info = describe_file(model_path);
    if (access(model_path.c_str(), R_OK) != 0) {
        g_engine.last_error = "model file is not readable: " + model_path + " (" + file_info + ")";
        return string_to_jstring(env, g_engine.last_error);
    }

    std::unique_lock<std::mutex> lock(g_engine.mutex);
    stop_generation_locked(lock);
    if (g_engine.model != nullptr) {
        llama_model_free(g_engine.model);
        g_engine.model = nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        g_engine.model_path.clear();
        g_engine.last_error = "failed to load model: " + model_path + " (" + file_info + ")";
        return string_to_jstring(env, g_engine.last_error);
    }

    g_engine.model = model;
    g_engine.model_path = model_path;
    g_engine.last_error.clear();
    return string_to_jstring(env, "model loaded: " + model_path + " (" + file_info + ")");
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_MainActivity_nativeGenerate(JNIEnv * env, jobject, jstring j_prompt, jobject callback) {
    ensure_backend();
    std::string prompt = jstring_to_string(env, j_prompt);
    if (prompt.empty()) {
        return string_to_jstring(env, "prompt is empty");
    }

    std::unique_lock<std::mutex> lock(g_engine.mutex);
    join_finished_worker_locked(lock);
    if (g_engine.model == nullptr) {
        return string_to_jstring(env, "model is not loaded");
    }
    if (g_engine.running) {
        return string_to_jstring(env, "generation is already running");
    }

    g_engine.cancel_requested.store(false);
    g_engine.running = true;
    g_engine.last_error.clear();
    jobject callback_ref = env->NewGlobalRef(callback);
    llama_model * model = g_engine.model;

    g_engine.worker = std::thread([callback_ref, model, prompt = std::move(prompt)] {
        run_generation(callback_ref, model, prompt);

        JNIEnv * env = nullptr;
        bool attached = false;
        if (g_vm != nullptr && g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env != nullptr) {
            env->DeleteGlobalRef(callback_ref);
        }
        if (attached) {
            g_vm->DetachCurrentThread();
        }
    });

    return string_to_jstring(env, "generation started");
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_MainActivity_nativeCancel(JNIEnv *, jobject) {
    g_engine.cancel_requested.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_MainActivity_nativeRelease(JNIEnv *, jobject) {
    std::unique_lock<std::mutex> lock(g_engine.mutex);
    stop_generation_locked(lock);
    if (g_engine.model != nullptr) {
        llama_model_free(g_engine.model);
        g_engine.model = nullptr;
    }
    g_engine.model_path.clear();
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_MainActivity_nativeGetEngineStats(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_engine.mutex);
    return string_to_jstring(env, stats_json_locked());
}
