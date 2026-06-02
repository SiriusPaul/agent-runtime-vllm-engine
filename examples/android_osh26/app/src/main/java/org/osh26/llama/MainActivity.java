package org.osh26.llama;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {
    private static final int REQUEST_IMPORT_MODEL = 1001;

    private TextView engineStatus;
    private TextView chatTranscript;
    private ScrollView chatScroll;
    private EditText modelPath;
    private EditText messageInput;
    private Button sendMessage;
    private final LlmHttpServer httpServer = new LlmHttpServer();
    private final StringBuilder conversationContext = new StringBuilder();
    private boolean generating = false;

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
        refreshStats();

        importModel.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_IMPORT_MODEL);
        });

        loadModel.setOnClickListener(v -> {
            appendSystemLine(LlamaNative.loadModel(modelPath.getText().toString()));
            refreshStats();
        });

        resetCache.setOnClickListener(v -> {
            LlamaNative.resetCache();
            conversationContext.setLength(0);
            chatTranscript.setText("");
            appendSystemLine("KV cache reset");
            refreshStats();
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
            new Thread(() -> {
                String status = LlamaNative.generateStream(generationPrompt, new LlamaNative.StreamCallback() {
                    @Override
                    public void onToken(String token) {
                        runOnUiThread(() -> {
                            chatTranscript.append(token);
                            scrollChatToBottom();
                        });
                    }

                    @Override
                    public void onComplete(String text, String finishReason) {
                        runOnUiThread(() -> {
                            generating = false;
                            sendMessage.setEnabled(true);
                            conversationContext.append("User: ").append(userMessage).append('\n')
                                    .append("Assistant: ").append(text).append("\n\n");
                            appendSystemLine("complete: " + finishReason);
                            refreshStats();
                        });
                    }

                    @Override
                    public void onError(String error) {
                        runOnUiThread(() -> {
                            generating = false;
                            sendMessage.setEnabled(true);
                            appendSystemLine("ERROR: " + error);
                            refreshStats();
                        });
                    }
                }, 128, 0.6f, 0.95f, 0xCAFE, false);
                runOnUiThread(() -> appendSystemLine(status));
            }, "osh26-ui-generate").start();
            refreshStats();
        });

        cancel.setOnClickListener(v -> {
            LlamaNative.cancel();
            generating = false;
            sendMessage.setEnabled(true);
            appendSystemLine("cancel requested");
            refreshStats();
        });

        startServer.setOnClickListener(v -> {
            appendSystemLine(httpServer.start());
            refreshStats();
        });

        stopServer.setOnClickListener(v -> {
            httpServer.stop();
            appendSystemLine("HTTP server stopped");
            refreshStats();
        });
    }

    @Override
    protected void onDestroy() {
        httpServer.stop();
        LlamaNative.release();
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

    private void refreshStats() {
        String status = LlamaNative.getEngineStats();
        status += "\nhttp_server_running=" + httpServer.isRunning() + ", port=" + LlmHttpServer.PORT;
        engineStatus.setText(status);
    }

    private String buildConversationPrompt(String userMessage) {
        if (conversationContext.length() == 0) {
            return userMessage;
        }
        return conversationContext.toString() + "User: " + userMessage;
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
