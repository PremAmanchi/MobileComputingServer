package com.example.project3;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.project3.ml.L1Model;

import org.opencv.android.OpenCVLoader;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    ImageView click_image_id;
    Context context;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button bt=(Button)findViewById(R.id.startserverBtn);
        click_image_id = findViewById(R.id.imageView);
        context = getApplicationContext();

        OpenCVLoader.initDebug();

        bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new start_server().execute();
            }
        });
    }

    class start_server extends AsyncTask<Void, Void, Integer> {
        @Override
        protected Integer doInBackground(Void... data) {
            try {
                L1Model model = L1Model.newInstance(MainActivity.this);
                AndroidWebServer androidWebServer = new AndroidWebServer(model, click_image_id, context);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            Toast.makeText(getBaseContext(),"Client Started",Toast.LENGTH_LONG).show();
        }
    }
}
