from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)

# ------------------------------
# Load Pretrained Models
# ------------------------------
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')
autoencoder = load_model("models/autoencoder.h5")  # trained model path
threshold = 0.02  # set after testing reconstruction error distribution

# ------------------------------
# Video Stream Generator
# ------------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = webcam, or give path for file

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize frame for feature extraction
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Feature extraction
        features = feature_extractor.predict(img)
        reconstructed = autoencoder.predict(features)

        # Compute reconstruction error
        error = np.mean(np.square(features - reconstructed))

        # Display result
        label = "Normal"
        color = (0, 255, 0)
        if error > threshold:
            label = "⚠️ Anomaly Detected!"
            color = (0, 0, 255)

        cv2.putText(frame, f"{label} | Error: {error:.4f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
