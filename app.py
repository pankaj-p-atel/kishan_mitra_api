from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model("model_uncompiled.h5", compile=False)
class_names = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']  # Replace with your actual class names

def preprocess_image(img):
    img = img.resize((256, 256))  # Use the image size you trained with
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img = Image.open(file.stream)
    img_array = preprocess_image(img)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    return jsonify({
        "class": class_names[class_index],
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
