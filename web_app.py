from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# 🔥 Load model SAFELY (fixes your error)
try:
    model = tf.keras.models.load_model("gesture_model.keras", compile=False)
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

# Load labels
labels = list(np.load("label_map.npy", allow_pickle=True))


def preprocess(frame):
    roi = cv2.resize(frame, (64, 64))
    roi = roi / 255.0
    return np.expand_dims(roi, axis=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"})

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"})

    try:
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"error": "Invalid image"})

        processed = preprocess(img)
        preds = model.predict(processed)[0]

        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        gesture = labels[pred_class] if pred_class < len(labels) else "Unknown"

        return jsonify({
            "gesture": gesture,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔥 IMPORTANT for Render (port binding fix)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
