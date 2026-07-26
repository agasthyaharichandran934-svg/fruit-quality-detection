"""
Fruit Quality Detection - Flask Web App
Serves the upload interface shown in Chapter 6 (Result and Discussion) of the
report: file upload -> Predict button -> "Prediction: The fruit is GOOD/BAD"
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 100

# ----------------------------------------------------------------------
# Load the trained model and class labels (see predict.py / report 5.1.1)
# ----------------------------------------------------------------------
model = tf.keras.models.load_model("fruit_classifier_model.h5")

class_labels = [
    "Good Orange", "Bad Orange",
    "Good Apple", "Bad Apple",
    "Good Pomegranate", "Bad Pomegranate"
]


def predict_fruit(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Error", 0.0

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    fruit_quality = "Bad" if "Bad" in class_labels[predicted_class] else "Good"
    return fruit_quality, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            fruit_quality, conf = predict_fruit(save_path)
            result = fruit_quality
            confidence = f"{conf:.2f}"
            image_path = save_path.replace("static/", "")

    return render_template(
        "index.html", result=result, confidence=confidence, image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)
