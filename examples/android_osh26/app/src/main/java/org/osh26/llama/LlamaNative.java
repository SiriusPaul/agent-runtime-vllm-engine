package org.osh26.llama;

public final class LlamaNative {
    static {
        System.loadLibrary("llama_osh26");
    }

    private LlamaNative() {
    }

    public interface StreamCallback {
        void onToken(String token);
        void onComplete(String text, String finishReason);
        void onError(String error);
    }

    public static native String loadModel(String modelPath);
    public static native String generateBlockingJson(String prompt, int maxTokens, float temperature, float topP, int seed, boolean thinking);
    public static native String generateStream(String prompt, StreamCallback callback, int maxTokens, float temperature, float topP, int seed, boolean thinking);
    public static native void cancel();
    public static native void release();
    public static native String getEngineStats();
}
