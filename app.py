from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
import os
import sys

app = Flask(__name__)

# ------------------------------
# Load Pretrained Models
# ------------------------------
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')

# Check if model file exists before loading
model_path = "models/autoencoder.h5"
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at '{model_path}'")
    print("\nPlease train the model first by running:")
    print("  python train_model.py")
    print("\nThis will create the required model file.")
    sys.exit(1)

if not os.path.exists("models"):
    print("ERROR: 'models' directory not found. Creating it...")
    os.makedirs("models")
    print(f"ERROR: Model file not found at '{model_path}'")
    print("\nPlease train the model first by running:")
    print("  python train_model.py")
    sys.exit(1)

try:
    autoencoder = load_model(model_path)  # trained model path
except Exception as e:
    print(f"ERROR: Failed to load model from '{model_path}'")
    print(f"Error details: {str(e)}")
    print("\nThe model file may be corrupted. Please retrain by running:")
    print("  python train_model.py")
    sys.exit(1)

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

        # Prepare the frame for the autoencoder (must match training input shape)
        # Resize to (224,224), convert BGR->RGB, normalize to [0,1], add batch dim
        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_img = frame_rgb.astype('float32') / 255.0
        input_batch = np.expand_dims(input_img, axis=0)  # shape: (1, 224, 224, 3)

        try:
            reconstructed_batch = autoencoder.predict(input_batch)
        except Exception as e:
            # Log and skip this frame to avoid breaking the streamed response
            print(f"Autoencoder predict error — skipping frame: {e}")
            continue

        # Remove batch dimension and convert back to BGR uint8 for display/comparison
        reconstructed = np.squeeze(reconstructed_batch, axis=0)
        reconstructed_bgr = cv2.cvtColor((reconstructed * 255).astype('uint8'), cv2.COLOR_RGB2BGR)

        # Compute difference / anomaly map if needed using the resized frame
        diff = cv2.absdiff(frame_resized, reconstructed_bgr)

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
