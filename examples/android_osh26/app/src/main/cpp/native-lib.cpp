#include <jni.h>

#include "osh26_engine.h"

#include <string>

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

jstring string_to_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
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
        jstring payload = env->NewStringUTF(text.c_str());
        jstring reason = env->NewStringUTF(finish_reason.c_str());
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
    osh26::GenerateResult result = osh26::engine().generate(
            jstring_to_string(env, j_prompt),
            options_from_args(max_tokens, temperature, top_p, seed, thinking),
            [callback_ref](const std::string & token) {
                call_stream_callback(callback_ref, "onToken", token);
            });

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
Java_org_osh26_llama_LlamaNative_release(JNIEnv *, jclass) {
    osh26::engine().release();
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_LlamaNative_getEngineStats(JNIEnv * env, jclass) {
    return string_to_jstring(env, osh26::engine().stats_json());
}
