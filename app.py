
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained CNN model once when the app starts
model = tf.keras.models.load_model("mnist_cnn.keras")

print("CNN model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the drawing sent by the browser
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No drawing received"}), 400

        # Remove the Base64 image prefix
        image_data = data["image"].split(",", 1)[-1]

        # Decode the image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # Create a black background and preserve white strokes
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

        # Convert white drawing on black background to grayscale
        image = image.convert("L")

        # Resize the drawing to MNIST dimensions
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert image into a NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values from 0-255 to 0-1
        image_array = image_array / 255.0

        # Add channel and batch dimensions
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict the digit
        predictions = model.predict(image_array, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return jsonify({
            "prediction": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": [
                round(float(value) * 100, 2)
                for value in predictions
            ]
        })

    except Exception as error:
        print("Prediction error:", error)
        return jsonify({
            "error": "Could not process the drawing. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
