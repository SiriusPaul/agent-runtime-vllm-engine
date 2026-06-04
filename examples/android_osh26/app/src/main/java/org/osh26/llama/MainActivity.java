package org.osh26.llama;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int REQUEST_IMPORT_MODEL = 1001;
    private static final String TAG = "OSH26Main";

    private TextView engineStatus;
    private TextView chatTranscript;
    private ScrollView chatScroll;
    private EditText modelPath;
    private EditText messageInput;
    private Button sendMessage;
    private final LlmHttpServer httpServer = new LlmHttpServer();
    private final ArrayDeque<String> conversationTurns = new ArrayDeque<>();
    private int conversationContextChars = 0;
    private final ExecutorService nativeExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "osh26-native-control");
        thread.setDaemon(true);
        return thread;
    });
    private final ExecutorService generationExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "osh26-generation");
        thread.setDaemon(true);
        return thread;
    });
    private boolean generating = false;
    private boolean loadingModel = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        engineStatus = findViewById(R.id.engine_status);
        chatTranscript = findViewById(R.id.chat_transcript);
        chatScroll = findViewById(R.id.chat_scroll);
        modelPath = findViewById(R.id.model_path);
        messageInput = findViewById(R.id.message_input);
        Button loadModel = findViewById(R.id.load_model);
        Button importModel = findViewById(R.id.import_model);
        Button resetCache = findViewById(R.id.reset_cache);
        sendMessage = findViewById(R.id.send_message);
        Button cancel = findViewById(R.id.cancel);
        Button startServer = findViewById(R.id.start_server);
        Button stopServer = findViewById(R.id.stop_server);

        File modelDir = new File(getFilesDir(), "models");
        modelPath.setText(new File(modelDir, "qwen3-0.6b.gguf").getAbsolutePath());
        appendSystemLine(httpServer.start());
        refreshStatsAsync();

        importModel.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_IMPORT_MODEL);
        });

        loadModel.setOnClickListener(v -> {
            if (loadingModel) {
                return;
            }
            final String path = modelPath.getText().toString();
            loadingModel = true;
            loadModel.setEnabled(false);
            appendSystemLine("loading model: " + path);
            nativeExecutor.execute(() -> {
                try {
                    final String status = LlamaNative.loadModel(path);
                    runOnUiThread(() -> {
                        loadingModel = false;
                        loadModel.setEnabled(true);
                        appendSystemLine(status);
                        refreshStatsAsync();
                    });
                } catch (Throwable t) {
                    Log.e(TAG, "loadModel failed", t);
                    runOnUiThread(() -> {
                        loadingModel = false;
                        loadModel.setEnabled(true);
                        appendSystemLine("load failed: " + t.getMessage());
                        refreshStatsAsync();
                    });
                }
            });
        });

        resetCache.setOnClickListener(v -> {
            clearConversationContext();
            chatTranscript.setText("");
            appendSystemLine("resetting KV cache...");
            nativeExecutor.execute(() -> {
                try {
                    LlamaNative.resetCache();
                    runOnUiThread(() -> {
                        appendSystemLine("KV cache reset");
                        refreshStatsAsync();
                    });
                } catch (Throwable t) {
                    Log.e(TAG, "resetCache failed", t);
                    runOnUiThread(() -> {
                        appendSystemLine("reset failed: " + t.getMessage());
                        refreshStatsAsync();
                    });
                }
            });
        });

        sendMessage.setOnClickListener(v -> {
            if (generating) {
                return;
            }
            String userMessage = messageInput.getText().toString().trim();
            if (userMessage.isEmpty()) {
                return;
            }
            messageInput.setText("");
            appendChatLine("User", userMessage);
            String generationPrompt = buildConversationPrompt(userMessage);
            generating = true;
            sendMessage.setEnabled(false);
            Log.i(TAG, "generate request: promptChars=" + generationPrompt.length()
                    + ", historyChars=" + conversationContextChars);
            appendSystemLine("generation: tokenizing prompt");
            appendSystemLine("generation: prefix cache check");
            appendSystemLine("generation: prefill in progress");
            generationExecutor.execute(() -> {
                try {
                    final boolean[] firstTokenSeen = new boolean[] { false };
                    String status = LlamaNative.generateStream(generationPrompt, new LlamaNative.StreamCallback() {
                        @Override
                        public void onToken(String token) {
                            runOnUiThread(() -> {
                                if (!firstTokenSeen[0]) {
                                    firstTokenSeen[0] = true;
                                    appendSystemLine("generation: decoding");
                                }
                                chatTranscript.append(token);
                                scrollChatToBottom();
                            });
                        }

                        @Override
                        public void onComplete(String text, String finishReason) {
                            runOnUiThread(() -> {
                                generating = false;
                                sendMessage.setEnabled(true);
                                appendConversationTurn(userMessage, text);
                                appendSystemLine("complete: " + finishReason);
                                refreshStatsAsync();
                            });
                        }

                        @Override
                        public void onError(String error) {
                            runOnUiThread(() -> {
                                generating = false;
                                sendMessage.setEnabled(true);
                                appendSystemLine("ERROR: " + error);
                                refreshStatsAsync();
                            });
                        }
                    }, 128, 0.6f, 0.95f, 0xCAFE, false);
                    runOnUiThread(() -> appendSystemLine(status));
                } catch (Throwable t) {
                    Log.e(TAG, "generateStream failed", t);
                    runOnUiThread(() -> {
                        generating = false;
                        sendMessage.setEnabled(true);
                        appendSystemLine("ERROR: " + t.getMessage());
                        refreshStatsAsync();
                    });
                }
            });
            refreshStatsAsync();
        });

        cancel.setOnClickListener(v -> {
            LlamaNative.cancel();
            generating = false;
            sendMessage.setEnabled(true);
            appendSystemLine("cancel requested");
            refreshStatsAsync();
        });

        startServer.setOnClickListener(v -> {
            appendSystemLine(httpServer.start());
            refreshStatsAsync();
        });

        stopServer.setOnClickListener(v -> {
            httpServer.stop();
            appendSystemLine("HTTP server stopped");
            refreshStatsAsync();
        });
    }

    @Override
    protected void onDestroy() {
        httpServer.stop();
        LlamaNative.cancel();
        nativeExecutor.execute(() -> {
            try {
                LlamaNative.release();
            } catch (Throwable t) {
                Log.e(TAG, "release failed", t);
            }
        });
        nativeExecutor.shutdown();
        generationExecutor.shutdown();
        super.onDestroy();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_IMPORT_MODEL || resultCode != RESULT_OK || data == null) {
            return;
        }

        Uri uri = data.getData();
        if (uri == null) {
            appendSystemLine("import failed: empty uri");
            return;
        }

        try {
            File modelDir = new File(getFilesDir(), "models");
            if (!modelDir.exists() && !modelDir.mkdirs()) {
                appendSystemLine("import failed: cannot create " + modelDir.getAbsolutePath());
                return;
            }

            String name = queryDisplayName(uri);
            if (name == null || name.trim().isEmpty()) {
                name = "model.gguf";
            }
            File target = new File(modelDir, name);

            try (InputStream input = getContentResolver().openInputStream(uri);
                 FileOutputStream output = new FileOutputStream(target)) {
                if (input == null) {
                    appendSystemLine("import failed: cannot open input stream");
                    return;
                }

                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = input.read(buffer)) >= 0) {
                    output.write(buffer, 0, n);
                }
            }

            modelPath.setText(target.getAbsolutePath());
            appendSystemLine("imported model: " + target.getAbsolutePath());
        } catch (Exception e) {
            appendSystemLine("import failed: " + e.getMessage());
        }
    }

    private String queryDisplayName(Uri uri) {
        try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        }
        return null;
    }

    private void refreshStatsAsync() {
        if (nativeExecutor.isShutdown()) {
            return;
        }
        nativeExecutor.execute(() -> {
            try {
                final String status = LlamaNative.getEngineStats()
                        + "\nhttp_server_running=" + httpServer.isRunning()
                        + ", port=" + LlmHttpServer.PORT;
                runOnUiThread(() -> engineStatus.setText(status));
            } catch (Throwable t) {
                Log.e(TAG, "refreshStats failed", t);
            }
        });
    }

    private String buildConversationPrompt(String userMessage) {
        StringBuilder prompt = new StringBuilder(Math.max(64, conversationContextChars + userMessage.length() + 32));
        for (String turn : conversationTurns) {
            prompt.append(turn);
        }
        prompt.append("User: ").append(userMessage).append('\n');
        prompt.append("Assistant: ");
        return prompt.toString();
    }

    private void appendConversationTurn(String userMessage, String assistantText) {
        String turn = "User: " + userMessage + '\n'
                + "Assistant: " + assistantText + "\n\n";
        conversationTurns.addLast(turn);
        conversationContextChars += turn.length();
    }

    private void clearConversationContext() {
        conversationTurns.clear();
        conversationContextChars = 0;
    }

    private void appendChatLine(String role, String text) {
        if (chatTranscript.length() > 0) {
            chatTranscript.append("\n");
        }
        chatTranscript.append(role + ": " + text + "\nAssistant: ");
        scrollChatToBottom();
    }

    private void appendSystemLine(String text) {
        if (chatTranscript.length() > 0) {
            chatTranscript.append("\n");
        }
        chatTranscript.append("[status] " + text + "\n");
        scrollChatToBottom();
    }

    private void scrollChatToBottom() {
        chatScroll.post(() -> chatScroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

}
