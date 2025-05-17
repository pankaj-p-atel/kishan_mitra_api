from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
from PIL import Image

app = Flask(__name__)

# Load model
try:
    model = load_model("potatoes.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)

# Class names corresponding to model output
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Preprocess incoming image
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return "Potato Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream)
        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            "class": class_names[class_index],
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
