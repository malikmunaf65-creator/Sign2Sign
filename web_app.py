from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("gesture_model.keras")
labels = list(np.load("label_map.npy", allow_pickle=True))


def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = cv2.imdecode(
                np.frombuffer(file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )

            processed = preprocess_image(img)
            preds = model.predict(processed)[0]

            pred_class = np.argmax(preds)
            confidence = float(np.max(preds))

            if pred_class < len(labels):
                prediction = labels[pred_class]
            else:
                prediction = "Unknown"

            confidence = f"{confidence*100:.2f}%"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
