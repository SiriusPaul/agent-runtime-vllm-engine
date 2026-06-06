package org.osh26.llama;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public final class LlmHttpServer {
    public static final int PORT = 8000;

    private final ExecutorService clients = Executors.newCachedThreadPool();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private ServerSocket serverSocket;
    private Thread acceptThread;

    public synchronized String start() {
        if (running.get()) {
            return "HTTP server already running on 127.0.0.1:" + PORT;
        }
        try {
            serverSocket = new ServerSocket(PORT, 16, InetAddress.getByName("127.0.0.1"));
            running.set(true);
            acceptThread = new Thread(this::acceptLoop, "osh26-http-accept");
            acceptThread.start();
            return "HTTP server listening on 127.0.0.1:" + PORT;
        } catch (IOException e) {
            running.set(false);
            return "failed to start HTTP server: " + e.getMessage();
        }
    }

    public synchronized void stop() {
        running.set(false);
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
            serverSocket = null;
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    private void acceptLoop() {
        while (running.get()) {
            try {
                Socket socket = serverSocket.accept();
                clients.execute(() -> handle(socket));
            } catch (IOException e) {
                if (running.get()) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void handle(Socket socket) {
        try (Socket ignored = socket;
             InputStream input = socket.getInputStream();
             OutputStream output = socket.getOutputStream()) {
            try {
                HttpRequest request = readRequest(input);
                if (request.requestLine == null || request.requestLine.trim().isEmpty()) {
                    return;
                }

                String[] parts = request.requestLine.split(" ");
                if (parts.length < 2) {
                    writeJson(output, 400, errorJson("bad_request", "invalid request line"));
                    return;
                }

                route(parts[0], parts[1], request.body, output);
            } catch (Throwable t) {
                Log.e("OSH26HTTP", "request failed: " + t.getMessage(), t);
                try {
                    writeJson(output, 500, errorJson("internal_error", t.toString()));
                } catch (Exception ignored2) {
                }
            }
        } catch (IOException e) {
            Log.e("OSH26HTTP", "socket failed: " + e.getMessage(), e);
        }
    }

    private HttpRequest readRequest(InputStream input) throws IOException {
        ByteArrayOutputStream headerBytes = new ByteArrayOutputStream(512);
        int matched = 0;
        int b;
        while ((b = input.read()) >= 0) {
            headerBytes.write(b);
            if ((matched == 0 && b == '\r')
                    || (matched == 1 && b == '\n')
                    || (matched == 2 && b == '\r')
                    || (matched == 3 && b == '\n')) {
                matched++;
                if (matched == 4) {
                    break;
                }
            } else {
                matched = b == '\r' ? 1 : 0;
            }
        }

        if (headerBytes.size() == 0) {
            return new HttpRequest(null, "");
        }

        String header = headerBytes.toString(StandardCharsets.ISO_8859_1.name());
        String[] lines = header.split("\r\n");
        String requestLine = lines.length > 0 ? lines[0] : null;
        int contentLength = 0;
        for (int i = 1; i < lines.length; i++) {
            String line = lines[i];
            int colon = line.indexOf(':');
            if (colon > 0 && "content-length".equals(line.substring(0, colon).trim().toLowerCase(Locale.US))) {
                contentLength = Integer.parseInt(line.substring(colon + 1).trim());
            }
        }

        byte[] bodyBytes = new byte[Math.max(0, contentLength)];
        int offset = 0;
        while (offset < bodyBytes.length) {
            int n = input.read(bodyBytes, offset, bodyBytes.length - offset);
            if (n < 0) {
                break;
            }
            offset += n;
        }
        String body = new String(bodyBytes, 0, offset, StandardCharsets.UTF_8);
        return new HttpRequest(requestLine, body);
    }

    private void route(String method, String path, String body, OutputStream writer) throws Exception {
        if ("GET".equals(method) && "/health".equals(path)) {
            JSONObject json = new JSONObject();
            json.put("status", "ok");
            json.put("port", PORT);
            json.put("engine", new JSONObject(LlamaNative.getEngineStats()));
            writeJson(writer, 200, json);
            return;
        }

        if ("GET".equals(method) && "/v1/models".equals(path)) {
            JSONObject model = new JSONObject();
            model.put("id", "local-gguf");
            model.put("object", "model");
            model.put("owned_by", "osh26");
            JSONObject json = new JSONObject();
            json.put("object", "list");
            json.put("data", new JSONArray().put(model));
            writeJson(writer, 200, json);
            return;
        }

        if ("POST".equals(method) && "/load_model".equals(path)) {
            JSONObject request = new JSONObject(body);
            String modelPath = request.optString("path", "");
            String backend = request.optString("backend", "auto");
            int nGpuLayers = request.optInt("n_gpu_layers", -1);
            boolean debugCorrectness = request.optBoolean("debug_correctness", false);
            Log.i("OSH26HTTP", "load_model path=" + modelPath + ", backend=" + backend + ", n_gpu_layers=" + nGpuLayers + ", debug_correctness=" + debugCorrectness);
            LlamaNative.configureBackend(backend, nGpuLayers);
            LlamaNative.setDebugCorrectness(debugCorrectness);
            JSONObject json = new JSONObject();
            json.put("result", LlamaNative.loadModel(modelPath));
            json.put("engine", new JSONObject(LlamaNative.getEngineStats()));
            Log.i("OSH26HTTP", "load_model completed");
            writeJson(writer, 200, json);
            return;
        }

        if ("POST".equals(method) && "/cancel".equals(path)) {
            LlamaNative.cancel();
            JSONObject json = new JSONObject();
            json.put("result", "cancel requested");
            json.put("engine", new JSONObject(LlamaNative.getEngineStats()));
            writeJson(writer, 200, json);
            return;
        }

        if ("POST".equals(method) && "/reset_cache".equals(path)) {
            LlamaNative.resetCache();
            JSONObject json = new JSONObject();
            json.put("result", "cache reset");
            json.put("engine", new JSONObject(LlamaNative.getEngineStats()));
            writeJson(writer, 200, json);
            return;
        }

        if ("POST".equals(method) && "/benchmark/quant_gemv".equals(path)) {
            writeJson(writer, 200, new JSONObject(LlamaNative.runQuantBenchmark()));
            return;
        }

        if ("POST".equals(method) && "/benchmark/quant_gemm".equals(path)) {
            writeJson(writer, 200, new JSONObject(LlamaNative.runQuantGemmBenchmark()));
            return;
        }

        if ("POST".equals(method) && "/v1/chat/completions".equals(path)) {
            handleChatCompletion(body, writer);
            return;
        }

        writeJson(writer, 404, errorJson("not_found", "unknown route"));
    }

    private void handleChatCompletion(String body, OutputStream writer) throws Exception {
        JSONObject request = new JSONObject(body);
        String prompt = messagesToPrompt(request.optJSONArray("messages"));
        int maxTokens = request.optInt("max_tokens", 128);
        float temperature = (float) request.optDouble("temperature", 0.6);
        float topP = (float) request.optDouble("top_p", 0.95);
        int seed = request.optInt("seed", 0xCAFE);
        boolean thinking = request.optBoolean("thinking", false);
        boolean stream = request.optBoolean("stream", false);

        if (stream) {
            writeSseHeaders(writer);
            String id = "chatcmpl-" + UUID.randomUUID();
            LlamaNative.generateStream(prompt, new LlamaNative.StreamCallback() {
                @Override
                public void onToken(String token) {
                    try {
                        JSONObject delta = new JSONObject();
                        delta.put("content", token);
                        JSONObject choice = new JSONObject();
                        choice.put("index", 0);
                        choice.put("delta", delta);
                        JSONObject chunk = new JSONObject();
                        chunk.put("id", id);
                        chunk.put("object", "chat.completion.chunk");
                        chunk.put("model", request.optString("model", "local-gguf"));
                        chunk.put("choices", new JSONArray().put(choice));
                        writeSseData(writer, chunk.toString());
                    } catch (Exception ignored) {
                    }
                }

                @Override
                public void onComplete(String text, String finishReason) {
                    try {
                        JSONObject choice = new JSONObject();
                        choice.put("index", 0);
                        choice.put("delta", new JSONObject());
                        choice.put("finish_reason", finishReason);
                        JSONObject chunk = new JSONObject();
                        chunk.put("id", id);
                        chunk.put("object", "chat.completion.chunk");
                        chunk.put("model", request.optString("model", "local-gguf"));
                        chunk.put("choices", new JSONArray().put(choice));
                        writeSseData(writer, chunk.toString());
                        writeSseData(writer, "[DONE]");
                    } catch (Exception ignored) {
                    }
                }

                @Override
                public void onError(String error) {
                    try {
                        writeSseData(writer, errorJson("generation_error", error).toString());
                        writeSseData(writer, "[DONE]");
                    } catch (Exception ignored) {
                    }
                }
            }, maxTokens, temperature, topP, seed, thinking);
            return;
        }

        JSONObject completion = new JSONObject(LlamaNative.generateBlockingJson(prompt, maxTokens, temperature, topP, seed, thinking));
        if (!completion.optBoolean("ok") && !"cancelled".equals(completion.optString("finish_reason"))) {
            writeJson(writer, 500, errorJson("generation_error", completion.optString("error")));
            return;
        }

        JSONObject message = new JSONObject();
        message.put("role", "assistant");
        message.put("content", completion.optString("text"));
        JSONObject choice = new JSONObject();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", completion.optString("finish_reason", "stop"));
        JSONObject json = new JSONObject();
        json.put("id", "chatcmpl-" + UUID.randomUUID());
        json.put("object", "chat.completion");
        json.put("model", request.optString("model", "local-gguf"));
        json.put("choices", new JSONArray().put(choice));
        JSONObject usage = new JSONObject();
        usage.put("completion_tokens", completion.optInt("decoded_tokens"));
        json.put("usage", usage);
        writeJson(writer, 200, json);
    }

    private String messagesToPrompt(JSONArray messages) {
        if (messages == null || messages.length() == 0) {
            return "";
        }
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < messages.length(); i++) {
            JSONObject message = messages.optJSONObject(i);
            if (message == null) {
                continue;
            }
            String content = message.optString("content", "");
            if (content.isEmpty()) {
                continue;
            }
            out.append(content).append('\n');
        }
        return out.toString().trim();
    }

    private JSONObject errorJson(String type, String message) throws Exception {
        JSONObject error = new JSONObject();
        error.put("type", type);
        error.put("message", message);
        return new JSONObject().put("error", error);
    }

    private void writeJson(OutputStream writer, int status, JSONObject json) throws IOException {
        byte[] bytes = json.toString().getBytes(StandardCharsets.UTF_8);
        String header = "HTTP/1.1 " + status + " " + reason(status) + "\r\n"
                + "Content-Type: application/json; charset=utf-8\r\n"
                + "Content-Length: " + bytes.length + "\r\n"
                + "Connection: close\r\n\r\n";
        writer.write(header.getBytes(StandardCharsets.US_ASCII));
        writer.write(bytes);
        writer.flush();
    }

    private void writeSseHeaders(OutputStream writer) throws IOException {
        writer.write(("HTTP/1.1 200 OK\r\n"
                + "Content-Type: text/event-stream; charset=utf-8\r\n"
                + "Cache-Control: no-cache\r\n"
                + "Connection: close\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
        writer.flush();
    }

    private synchronized void writeSseData(OutputStream writer, String data) throws IOException {
        writer.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        writer.flush();
    }

    private String reason(int status) {
        if (status == 200) {
            return "OK";
        }
        if (status == 400) {
            return "Bad Request";
        }
        if (status == 404) {
            return "Not Found";
        }
        return "Internal Server Error";
    }

    private static final class HttpRequest {
        final String requestLine;
        final String body;

        HttpRequest(String requestLine, String body) {
            this.requestLine = requestLine;
            this.body = body;
        }
    }
}
