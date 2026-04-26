from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("gesture_model.keras", compile=False)
labels = list(np.load("label_map.npy", allow_pickle=True))


def preprocess(frame):
    roi = cv2.resize(frame, (64, 64)) / 255.0
    return np.expand_dims(roi, axis=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file"})

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    processed = preprocess(img)
    preds = model.predict(processed)[0]

    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    gesture = labels[pred_class] if pred_class < len(labels) else "Unknown"

    return jsonify({
        "gesture": gesture,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
