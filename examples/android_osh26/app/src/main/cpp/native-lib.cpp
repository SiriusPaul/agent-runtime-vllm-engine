#include <jni.h>

#include "osh26_engine.h"
#include "osh26_vk_gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

JavaVM * g_vm = nullptr;

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

size_t utf8_valid_prefix_length(const std::string & value) {
    size_t i = 0;
    const size_t size = value.size();
    while (i < size) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c < 0x80) {
            ++i;
            continue;
        }

        size_t expected = 0;
        uint32_t codepoint = 0;
        if ((c >> 5) == 0x6) {
            expected = 2;
            codepoint = c & 0x1F;
        } else if ((c >> 4) == 0xE) {
            expected = 3;
            codepoint = c & 0x0F;
        } else if ((c >> 3) == 0x1E) {
            expected = 4;
            codepoint = c & 0x07;
        } else {
            break;
        }

        if (i + expected > size) {
            break;
        }

        bool valid = true;
        for (size_t j = 1; j < expected; ++j) {
            const unsigned char d = static_cast<unsigned char>(value[i + j]);
            if ((d >> 6) != 0x2) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (d & 0x3F);
        }
        if (!valid) {
            break;
        }

        if ((expected == 2 && codepoint < 0x80)
                || (expected == 3 && codepoint < 0x800)
                || (expected == 4 && codepoint < 0x10000)
                || (codepoint >= 0xD800 && codepoint <= 0xDFFF)
                || codepoint > 0x10FFFF) {
            break;
        }

        i += expected;
    }
    return i;
}

jstring string_to_jstring(JNIEnv * env, const std::string & value) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(value.size()));
    if (bytes == nullptr) {
        return nullptr;
    }
    if (!value.empty()) {
        env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(value.size()), reinterpret_cast<const jbyte *>(value.data()));
    }

    jclass charset_class = env->FindClass("java/nio/charset/StandardCharsets");
    if (charset_class == nullptr) {
        env->DeleteLocalRef(bytes);
        return nullptr;
    }
    jfieldID utf8_field = env->GetStaticFieldID(charset_class, "UTF_8", "Ljava/nio/charset/Charset;");
    if (utf8_field == nullptr) {
        env->DeleteLocalRef(charset_class);
        env->DeleteLocalRef(bytes);
        return nullptr;
    }
    jobject utf8_charset = env->GetStaticObjectField(charset_class, utf8_field);
    if (utf8_charset == nullptr) {
        env->DeleteLocalRef(charset_class);
        env->DeleteLocalRef(bytes);
        return nullptr;
    }

    jclass string_class = env->FindClass("java/lang/String");
    if (string_class == nullptr) {
        env->DeleteLocalRef(utf8_charset);
        env->DeleteLocalRef(charset_class);
        env->DeleteLocalRef(bytes);
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/nio/charset/Charset;)V");
    if (ctor == nullptr) {
        env->DeleteLocalRef(string_class);
        env->DeleteLocalRef(utf8_charset);
        env->DeleteLocalRef(charset_class);
        env->DeleteLocalRef(bytes);
        return nullptr;
    }

    jstring result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, utf8_charset));
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(utf8_charset);
    env->DeleteLocalRef(charset_class);
    env->DeleteLocalRef(bytes);
    return result;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

std::string result_to_json(const osh26::GenerateResult & result) {
    return std::string("{")
            + "\"ok\":" + (result.ok ? "true" : "false")
            + ",\"cancelled\":" + (result.cancelled ? "true" : "false")
            + ",\"finish_reason\":\"" + json_escape(result.finish_reason) + "\""
            + ",\"decoded_tokens\":" + std::to_string(result.decoded_tokens)
            + ",\"ttft_ms\":" + std::to_string(result.ttft_ms)
            + ",\"tokens_per_second\":" + std::to_string(result.tokens_per_second)
            + ",\"error\":\"" + json_escape(result.error) + "\""
            + ",\"token_ids\":\"" + json_escape(result.token_ids) + "\""
            + ",\"text\":\"" + json_escape(result.text) + "\""
            + "}";
}

void call_stream_callback(jobject callback, const char * method_name, const std::string & value) {
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
        jstring payload = string_to_jstring(env, value);
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

void call_complete_callback(jobject callback, const std::string & text, const std::string & finish_reason) {
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
    jmethodID method = env->GetMethodID(callback_class, "onComplete", "(Ljava/lang/String;Ljava/lang/String;)V");
    if (method != nullptr) {
        jstring payload = string_to_jstring(env, text);
        jstring reason = string_to_jstring(env, finish_reason);
        env->CallVoidMethod(callback, method, payload, reason);
        env->DeleteLocalRef(reason);
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

osh26::GenerateOptions options_from_args(jint max_tokens, jfloat temperature, jfloat top_p, jint seed, jboolean thinking) {
    osh26::GenerateOptions options;
    options.max_tokens = max_tokens > 0 ? max_tokens : 128;
    options.temperature = temperature > 0.0f ? temperature : 0.6f;
    options.top_p = top_p > 0.0f ? top_p : 0.95f;
    options.seed = seed > 0 ? (uint32_t) seed : 0xCAFE;
    options.enable_thinking = thinking == JNI_TRUE;
    return options;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_loadModel(JNIEnv * env, jclass, jstring j_model_path) {
    return string_to_jstring(env, osh26::engine().load_model(jstring_to_string(env, j_model_path)));
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_LlamaNative_configureBackend(JNIEnv * env, jclass, jstring j_mode, jint n_gpu_layers) {
    osh26::engine().configure_backend(jstring_to_string(env, j_mode), n_gpu_layers);
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_LlamaNative_setDebugCorrectness(JNIEnv *, jclass, jboolean enabled) {
    osh26::engine().set_debug_correctness(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_generateBlockingJson(
        JNIEnv * env, jclass, jstring j_prompt, jint max_tokens, jfloat temperature, jfloat top_p, jint seed, jboolean thinking) {
    osh26::GenerateResult result = osh26::engine().generate(
            jstring_to_string(env, j_prompt),
            options_from_args(max_tokens, temperature, top_p, seed, thinking),
            nullptr);
    return string_to_jstring(env, result_to_json(result));
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_generateStream(
        JNIEnv * env, jclass, jstring j_prompt, jobject callback, jint max_tokens, jfloat temperature, jfloat top_p, jint seed, jboolean thinking) {
    jobject callback_ref = env->NewGlobalRef(callback);
    std::string pending_stream_bytes;
    osh26::GenerateResult result = osh26::engine().generate(
            jstring_to_string(env, j_prompt),
            options_from_args(max_tokens, temperature, top_p, seed, thinking),
            [callback_ref, &pending_stream_bytes](const std::string & token) {
                pending_stream_bytes += token;
                const size_t emit_len = utf8_valid_prefix_length(pending_stream_bytes);
                if (emit_len > 0) {
                    call_stream_callback(callback_ref, "onToken", pending_stream_bytes.substr(0, emit_len));
                    pending_stream_bytes.erase(0, emit_len);
                }
            });

    if (!pending_stream_bytes.empty()) {
        call_stream_callback(callback_ref, "onToken", pending_stream_bytes);
        pending_stream_bytes.clear();
    }

    if (result.ok) {
        call_complete_callback(callback_ref, result.text, result.finish_reason);
    } else if (result.cancelled) {
        call_complete_callback(callback_ref, result.text, result.finish_reason);
    } else {
        call_stream_callback(callback_ref, "onError", result.error);
    }
    env->DeleteGlobalRef(callback_ref);
    return string_to_jstring(env, result.ok ? "generation complete" : "generation failed: " + result.error);
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_LlamaNative_cancel(JNIEnv *, jclass) {
    osh26::engine().cancel();
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_LlamaNative_resetCache(JNIEnv *, jclass) {
    osh26::engine().reset_cache();
}

extern "C" JNIEXPORT void JNICALL
Java_org_osh26_llama_LlamaNative_release(JNIEnv *, jclass) {
    osh26::engine().release();
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_getEngineStats(JNIEnv * env, jclass) {
    return string_to_jstring(env, osh26::engine().stats_json());
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_runQuantBenchmark(JNIEnv * env, jclass) {
    char json[4096];
    osh26_vk_gpu_quant_benchmark(json, sizeof(json));
    return string_to_jstring(env, json);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_runQuantGemmBenchmark(JNIEnv * env, jclass) {
    char json[4096];
    osh26_vk_gpu_quant_gemm_benchmark(json, sizeof(json));
    return string_to_jstring(env, json);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_runQ8GemmBenchmark(JNIEnv * env, jclass) {
    char json[8192];
    osh26_vk_gpu_q8_gemm_benchmark(json, sizeof(json));
    return string_to_jstring(env, json);
}
