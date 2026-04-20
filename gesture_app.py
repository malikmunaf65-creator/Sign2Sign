# gesture_app.py (with camera color/flip fix)

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty, BooleanProperty, NumericProperty

import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

try:
    model = tf.keras.models.load_model("gesture_model.keras")
    labels = list(np.load("label_map.npy", allow_pickle=True))
    model_loaded = True
    print(f"✅ Loaded model with gestures: {labels}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("ℹ️  You can still collect data. Train the model later with: python train_model.py")
    model_loaded = False
    labels = []

KV = '''
<GestureApp>:
    orientation: 'vertical'
    padding: 10
    spacing: 10
    BoxLayout:
        size_hint: 1, 0.6
        RelativeLayout:
            id: camera_container
            Camera:
                id: camera
                resolution: (640, 480)
                play: True
                allow_stretch: True
                keep_ratio: True
            Widget:
                canvas:
                    Color:
                        rgba: 0, 1, 0, 0.4
                    Line:
                        rectangle: self.x + self.width*0.25, self.y + self.height*0.25, self.width*0.5, self.height*0.5
                        width: 3
    BoxLayout:
        size_hint: 1, 0.4
        orientation: 'vertical'
        padding: 10
        spacing: 10
        Label:
            text: "Gesture Recognition"
            font_size: '24sp'
            bold: True
            size_hint_y: 0.15
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.4
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: "Current Gesture:"
                    font_size: '18sp'
                Label:
                    text: root.prediction_text
                    font_size: '28sp'
                    bold: True
                    color: root.prediction_color
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: "Confidence:"
                    font_size: '18sp'
                Label:
                    text: root.confidence_text
                    font_size: '22sp'
        BoxLayout:
            size_hint_y: 0.3
            orientation: 'vertical'
            Label:
                text: "Recent Gestures:"
                font_size: '16sp'
            Label:
                text: root.history_text
                font_size: '16sp'
        BoxLayout:
            size_hint_y: 0.15
            spacing: 10
            Button:
                text: "Start Prediction" if not root.predicting else "Stop Prediction"
                on_press: root.toggle_prediction()
                background_color: (0.2, 0.8, 0.2, 1) if root.predicting else (0.8, 0.2, 0.2, 1)
            Button:
                text: "Add New Gesture"
                on_press: root.show_add_gesture_popup()
                background_color: (0.2, 0.5, 0.8, 1)

<AddGesturePopup>:
    size_hint: 0.7, 0.4
    title: "Add New Gesture"
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        Label:
            text: "Enter gesture name (HELLO, YES, NO, etc.):"
            font_size: '16sp'
        TextInput:
            id: gesture_name
            multiline: False
            font_size: '20sp'
            hint_text: "Type gesture name here"
        BoxLayout:
            size_hint_y: 0.3
            spacing: 10
            Button:
                text: "Cancel"
                on_press: root.dismiss()
                background_color: (0.8, 0.2, 0.2, 1)
            Button:
                text: "Start Collection"
                on_press: root.save_gesture()
                background_color: (0.2, 0.8, 0.2, 1)

<TrainGesturePopup>:
    size_hint: 0.5, 0.35
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        padding: 15
        spacing: 8
        Label:
            text: root.status_text
            font_size: '18sp'
            bold: True
            color: (0.2, 1, 0.2, 1)
        ProgressBar:
            max: 100
            value: root.progress_value
            size_hint_y: 0.15
        Label:
            text: "👁 Watch the camera - vary your gesture!"
            font_size: '14sp'
            color: (1, 1, 0.3, 1)
        Button:
            text: "Cancel"
            size_hint_y: 0.25
            on_press: root.cancel_collection()
            background_color: (0.8, 0.2, 0.2, 1)
'''

Builder.load_string(KV)

class AddGesturePopup(Popup):
    def save_gesture(self):
        gesture_name = self.ids.gesture_name.text.strip().upper()
        if not gesture_name:
            print("❌ Please enter a gesture name!")
            return
        if gesture_name in labels:
            print(f"⚠️ Gesture '{gesture_name}' already exists. Adding more samples...")
        else:
            labels.append(gesture_name)
            print(f"✅ Added new gesture: {gesture_name}")
            np.save('label_map.npy', np.array(labels))
        self.dismiss()
        train_popup = TrainGesturePopup(gesture_name=gesture_name)
        train_popup.open()

class TrainGesturePopup(Popup):
    status_text = StringProperty("Collecting...")
    progress_value = NumericProperty(0)
    def __init__(self, gesture_name, **kwargs):
        self.gesture_name = gesture_name
        self.samples_collected = 0
        self.total_samples = 100
        self.is_collecting = False
        super().__init__(**kwargs)
        self.title = f"Collecting: {gesture_name}"
        self.status_text = f"{gesture_name}: 0/{self.total_samples}"
        self.progress_value = 0
    def on_open(self):
        self.is_collecting = True
        Clock.schedule_interval(self.capture_sample, 0.3)
    def cancel_collection(self):
        self.is_collecting = False
        Clock.unschedule(self.capture_sample)
        self.dismiss()
        print(f"⚠️ Collection cancelled for {self.gesture_name}")
        print(f"   Collected {self.samples_collected}/{self.total_samples} samples")
    def capture_sample(self, dt):
        if not self.is_collecting or self.samples_collected >= self.total_samples:
            Clock.unschedule(self.capture_sample)
            if self.samples_collected >= self.total_samples:
                self.dismiss()
                self.show_completion_message()
            return
        app = App.get_running_app()
        if hasattr(app, 'root') and hasattr(app.root, 'ids'):
            camera = app.root.ids.camera
            if camera and hasattr(camera, 'texture') and camera.texture:
                try:
                    buffer = camera.texture.pixels
                    frame = np.frombuffer(buffer, np.uint8)
                    frame = frame.reshape((camera.texture.height, camera.texture.width, 4))
                    frame = frame[:, :, :3]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 0)
                    h, w = frame.shape[:2]
                    roi_size = min(h, w) // 2
                    x1 = (w - roi_size) // 2
                    y1 = (h - roi_size) // 2
                    roi = frame[y1:y1 + roi_size, x1:x1 + roi_size]
                    roi_resized = cv2.resize(roi, (64, 64))
                    save_path = f"gesture_data/{self.gesture_name}"
                    os.makedirs(save_path, exist_ok=True)
                    existing_files = [f for f in os.listdir(save_path) if f.endswith('.npy')]
                    file_number = len(existing_files)
                    filename = f"{save_path}/{file_number}.npy"
                    np.save(filename, roi_resized)
                    self.samples_collected += 1
                    self.status_text = f"{self.gesture_name}: {self.samples_collected}/{self.total_samples}"
                    self.progress_value = (self.samples_collected / self.total_samples) * 100
                except Exception as e:
                    print(f"❌ Error capturing sample: {e}")
    def show_completion_message(self):
        print("\n" + "="*60)
        print(f"✅ Data collection complete for {self.gesture_name}!")
        print(f"📊 Collected {self.samples_collected} samples")
        print("="*60)
        print("\n🎯 NEXT STEPS:")
        print("1. Collect samples for other gestures (repeat for all 7 gestures)")
        print("2. When done, close the app and run: python train_model.py")
        print("3. After training, restart the app to test predictions!")
        print("="*60 + "\n")

class GestureApp(BoxLayout):
    prediction_text = StringProperty("Show your gesture")
    confidence_text = StringProperty("0%")
    prediction_color = ListProperty([1, 1, 1, 1])
    history_text = StringProperty("")
    predicting = BooleanProperty(False)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gesture_history = deque(maxlen=5)
    def toggle_prediction(self):
        if not model_loaded:
            self.prediction_text = "Model not loaded!"
            self.confidence_text = "0%"
            self.prediction_color = [1, 0, 0, 1]
            print("❌ Please train the model first: python train_model.py")
            return
        self.predicting = not self.predicting
        if self.predicting:
            print("🎥 Starting predictions...")
            Clock.schedule_interval(self.update_prediction, 1.0 / 5)
        else:
            print("⏸️ Stopped predictions")
            Clock.unschedule(self.update_prediction)
            self.prediction_text = "Stopped"
            self.confidence_text = "0%"
            self.prediction_color = [1, 0, 0, 1]
    def update_prediction(self, dt):
        if not self.predicting:
            return
        camera = self.ids.camera
        if camera and camera.texture:
            try:
                buffer = camera.texture.pixels
                frame = np.frombuffer(buffer, np.uint8)
                frame = frame.reshape((camera.texture.height, camera.texture.width, 4))
                frame = frame[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                h, w = frame.shape[:2]
                roi_size = min(h, w) // 2
                x1 = (w - roi_size) // 2
                y1 = (h - roi_size) // 2
                roi = frame[y1:y1 + roi_size, x1:x1 + roi_size]
                roi_resized = cv2.resize(roi, (64, 64))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)
                predictions = model.predict(roi_input, verbose=0)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)
                if len(labels) > predicted_class:
                    gesture_name = labels[predicted_class]
                else:
                    gesture_name = "Unknown"
                if confidence > 0.4:
                    self.prediction_text = gesture_name
                    self.confidence_text = f"{confidence * 100:.1f}%"
                    self.prediction_color = [0, 1, 0, 1]
                    self.gesture_history.append(f"{gesture_name} ({confidence * 100:.1f}%)")
                    self.history_text = "\n".join(self.gesture_history)
                else:
                    self.prediction_text = "Unknown"
                    self.confidence_text = f"{confidence * 100:.1f}%"
                    self.prediction_color = [1, 1, 0, 1]
            except Exception as e:
                print(f"❌ Prediction error: {e}")
    def show_add_gesture_popup(self):
        popup = AddGesturePopup()
        popup.open()

class GestureAppMain(App):
    def build(self):
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        return GestureApp()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 GESTURE RECOGNITION APP")
    print("="*60)
    print("📌 Instructions:")
    print("   1. Click 'Add New Gesture' to collect training data")
    print("   2. Collect 100 samples per gesture (7 gestures recommended)")
    print("   3. Close app and run: python train_model.py")
    print("   4. Restart app and click 'Start Prediction' to test!")
    print("="*60 + "\n")
    GestureAppMain().run()
