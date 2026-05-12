package org.osh26.llama;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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
             BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8))) {
            String requestLine = reader.readLine();
            if (requestLine == null || requestLine.trim().isEmpty()) {
                return;
            }

            String[] parts = requestLine.split(" ");
            if (parts.length < 2) {
                writeJson(writer, 400, errorJson("bad_request", "invalid request line"));
                return;
            }

            int contentLength = 0;
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                int colon = line.indexOf(':');
                if (colon > 0 && "content-length".equals(line.substring(0, colon).trim().toLowerCase(Locale.US))) {
                    contentLength = Integer.parseInt(line.substring(colon + 1).trim());
                }
            }

            String body = "";
            if (contentLength > 0) {
                char[] buffer = new char[contentLength];
                int read = 0;
                while (read < contentLength) {
                    int n = reader.read(buffer, read, contentLength - read);
                    if (n < 0) {
                        break;
                    }
                    read += n;
                }
                body = new String(buffer, 0, read);
            }

            route(parts[0], parts[1], body, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void route(String method, String path, String body, BufferedWriter writer) throws Exception {
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
            JSONObject json = new JSONObject();
            json.put("result", LlamaNative.loadModel(modelPath));
            json.put("engine", new JSONObject(LlamaNative.getEngineStats()));
            writeJson(writer, 200, json);
            return;
        }

        if ("POST".equals(method) && "/v1/chat/completions".equals(path)) {
            handleChatCompletion(body, writer);
            return;
        }

        writeJson(writer, 404, errorJson("not_found", "unknown route"));
    }

    private void handleChatCompletion(String body, BufferedWriter writer) throws Exception {
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
            String role = message.optString("role", "user");
            String content = message.optString("content", "");
            if (content.isEmpty()) {
                continue;
            }
            out.append(role).append(": ").append(content).append('\n');
        }
        return out.toString().trim();
    }

    private JSONObject errorJson(String type, String message) throws Exception {
        JSONObject error = new JSONObject();
        error.put("type", type);
        error.put("message", message);
        return new JSONObject().put("error", error);
    }

    private void writeJson(BufferedWriter writer, int status, JSONObject json) throws IOException {
        byte[] bytes = json.toString().getBytes(StandardCharsets.UTF_8);
        writer.write("HTTP/1.1 " + status + " " + reason(status) + "\r\n");
        writer.write("Content-Type: application/json; charset=utf-8\r\n");
        writer.write("Content-Length: " + bytes.length + "\r\n");
        writer.write("Connection: close\r\n\r\n");
        writer.write(json.toString());
        writer.flush();
    }

    private void writeSseHeaders(BufferedWriter writer) throws IOException {
        writer.write("HTTP/1.1 200 OK\r\n");
        writer.write("Content-Type: text/event-stream; charset=utf-8\r\n");
        writer.write("Cache-Control: no-cache\r\n");
        writer.write("Connection: close\r\n\r\n");
        writer.flush();
    }

    private synchronized void writeSseData(BufferedWriter writer, String data) throws IOException {
        writer.write("data: ");
        writer.write(data);
        writer.write("\n\n");
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
}
