package com.example.project3;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.project3.ml.L1Model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import fi.iki.elonen.NanoHTTPD;

public class AndroidWebServer extends NanoHTTPD {
    private static L1Model model;
    private static ImageView imageView;
    private static Context context;
    private Map<String, String> files = new HashMap<String, String>();
    Mat q = new Mat();

    public AndroidWebServer(L1Model _model, ImageView _imageView, Context _context) throws IOException {
        super(8080);
        start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
        model = _model;
        imageView = _imageView;
        context = _context;
        System.out.println("\nRunning! Point your browsers to http://localhost:8080/ \n");
    }

    public static void main(String[] args) {
        try {
            new AndroidWebServer(model, imageView, context);
        } catch (IOException ioe) {
            System.err.println("Couldn't start server:\n" + ioe);
        }
    }

    @Override
    public Response serve(IHTTPSession session) {
        StringBuilder responseMsg = new StringBuilder();

        try {
            session.parseBody(this.files);
            JSONObject reader = new JSONObject(files.get("postData"));
            String  b64_encoded_image  = new String(reader.getString("encoded_image"));
            b64_encoded_image = b64_encoded_image.replace("%20", "\n");
            byte[] bitmapdata = Base64.decode(b64_encoded_image, Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(bitmapdata, 0, bitmapdata.length);

            Handler handler = new Handler(Looper.getMainLooper());

            handler.post(new Runnable() {
                @Override
                public void run() {
                    imageView.setImageBitmap(bitmap);
                    Toast.makeText(context, "Received Image, running Tensorflow model", Toast.LENGTH_LONG).show();
                }
            });

            Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(bmp32, q);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 14, 14}, DataType.FLOAT32);
            Bitmap q_bitmap = Bitmap.createBitmap(14, 14, Bitmap.Config.ARGB_8888);;
            Utils.matToBitmap(q, q_bitmap);
            TensorImage image = new TensorImage(DataType.FLOAT32);
            image.load(q_bitmap);
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new TransformToGrayscaleOp())
                            .build();
            imageProcessor.process(image);
            ByteBuffer byteBuffer = image.getBuffer();
            Log.d("bytebuffer", byteBuffer.toString());
            inputFeature0.loadBuffer(byteBuffer);
            Log.d("bytebuffer", inputFeature0.getBuffer().toString());

            L1Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output = outputFeature0.getFloatArray();

            model.close();

            responseMsg.append(Arrays.toString(output));

        } catch (Exception e) {
            responseMsg.append(e.getMessage());
        }
        return newFixedLengthResponse(responseMsg.toString());
    }
}
