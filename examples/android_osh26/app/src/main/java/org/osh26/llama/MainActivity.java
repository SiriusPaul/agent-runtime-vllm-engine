package org.osh26.llama;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {
    static {
        System.loadLibrary("llama_osh26");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView engineStatus = findViewById(R.id.engine_status);
        engineStatus.setText(nativeGetEngineStats());
    }

    private native String nativeGetEngineStats();
}
