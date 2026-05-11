#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_org_osh26_llama_MainActivity_nativeGetEngineStats(
        JNIEnv* env,
        jobject /* this */) {
    std::string stats = "OSH26 mock engine ready\nmodel class: small GGUF, target <= 3B\nbackend: JNI native stub";
    return env->NewStringUTF(stats.c_str());
}
