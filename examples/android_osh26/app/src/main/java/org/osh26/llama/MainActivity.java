package org.osh26.llama;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {
    private static final int REQUEST_IMPORT_MODEL = 1001;

    private TextView engineStatus;
    private TextView output;
    private EditText modelPath;
    private EditText prompt;

    static {
        System.loadLibrary("llama_osh26");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        engineStatus = findViewById(R.id.engine_status);
        output = findViewById(R.id.output);
        modelPath = findViewById(R.id.model_path);
        prompt = findViewById(R.id.prompt);
        Button loadModel = findViewById(R.id.load_model);
        Button importModel = findViewById(R.id.import_model);
        Button generate = findViewById(R.id.generate);
        Button cancel = findViewById(R.id.cancel);

        File modelDir = new File(getFilesDir(), "models");
        modelPath.setText(new File(modelDir, "qwen3-0.6b.gguf").getAbsolutePath());
        refreshStats();

        importModel.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_IMPORT_MODEL);
        });

        loadModel.setOnClickListener(v -> {
            appendLine(nativeLoadModel(modelPath.getText().toString()));
            refreshStats();
        });

        generate.setOnClickListener(v -> {
            output.setText("");
            appendLine(nativeGenerate(prompt.getText().toString(), new TokenCallback() {
                @Override
                public void onToken(String token) {
                    runOnUiThread(() -> output.append(token));
                }

                @Override
                public void onComplete(String stats) {
                    runOnUiThread(() -> {
                        appendLine("\n" + stats);
                        refreshStats();
                    });
                }

                @Override
                public void onError(String error) {
                    runOnUiThread(() -> {
                        appendLine("\nERROR: " + error);
                        refreshStats();
                    });
                }
            }));
            refreshStats();
        });

        cancel.setOnClickListener(v -> {
            nativeCancel();
            appendLine("\ncancel requested");
            refreshStats();
        });
    }

    @Override
    protected void onDestroy() {
        nativeRelease();
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
            appendLine("import failed: empty uri");
            return;
        }

        try {
            File modelDir = new File(getFilesDir(), "models");
            if (!modelDir.exists() && !modelDir.mkdirs()) {
                appendLine("import failed: cannot create " + modelDir.getAbsolutePath());
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
                    appendLine("import failed: cannot open input stream");
                    return;
                }

                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = input.read(buffer)) >= 0) {
                    output.write(buffer, 0, n);
                }
            }

            modelPath.setText(target.getAbsolutePath());
            appendLine("imported model: " + target.getAbsolutePath());
        } catch (Exception e) {
            appendLine("import failed: " + e.getMessage());
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
        engineStatus.setText(nativeGetEngineStats());
    }

    private void appendLine(String text) {
        output.append(text);
        output.append("\n");
    }

    public interface TokenCallback {
        void onToken(String token);
        void onComplete(String stats);
        void onError(String error);
    }

    private native String nativeLoadModel(String modelPath);
    private native String nativeGenerate(String prompt, TokenCallback callback);
    private native void nativeCancel();
    private native void nativeRelease();
    private native String nativeGetEngineStats();
}
