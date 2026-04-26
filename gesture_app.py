# gesture_app.py — Sign2Sign Premium UI
# Upload this file to GitHub replacing the existing gesture_app.py

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty, BooleanProperty, NumericProperty
from kivy.graphics import Color, RoundedRectangle, Rectangle, Line, Ellipse
from kivy.animation import Animation
import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# ── WINDOW SETUP ────────────────────────────────────────────
Window.clearcolor = (0.02, 0.05, 0.04, 1)   # #050d0a — deep obsidian green
Window.size = (900, 660)

# ── MODEL LOAD ───────────────────────────────────────────────
try:
    model = tf.keras.models.load_model("gesture_model.keras")
    labels = list(np.load("label_map.npy", allow_pickle=True))
    model_loaded = True
    print(f"✅ Model loaded | Gestures: {labels}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model_loaded = False
    labels = []

# ── KV STRING ────────────────────────────────────────────────
KV = '''
#:import Window kivy.core.window.Window
#:import Clock kivy.clock.Clock

<RoundedButton@Button>:
    background_color: 0, 0, 0, 0
    background_normal: ''
    canvas.before:
        Color:
            rgba: self._bg_color if not self.state == 'down' else [c * 0.8 for c in self._bg_color]
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self._radius]
    _bg_color: (0, 0.9, 0.63, 1)
    _radius: 12

<GlassCard@BoxLayout>:
    canvas.before:
        Color:
            rgba: 0, 0.9, 0.63, 0.06
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [18]
        Color:
            rgba: 0, 0.9, 0.63, 0.13
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, 18
            width: 1

<GlassCardAmber@BoxLayout>:
    canvas.before:
        Color:
            rgba: 0.96, 0.62, 0.04, 0.06
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [18]
        Color:
            rgba: 0.96, 0.62, 0.04, 0.13
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, 18
            width: 1

<Sign2SignLayout>:
    orientation: 'horizontal'
    spacing: 0

    # ── LEFT SIDEBAR ──────────────────────────────
    BoxLayout:
        size_hint_x: 0.22
        orientation: 'vertical'
        padding: [16, 20, 16, 20]
        spacing: 6
        canvas.before:
            Color:
                rgba: 0.02, 0.06, 0.04, 1
            Rectangle:
                pos: self.pos
                size: self.size
            Color:
                rgba: 0, 0.9, 0.63, 0.08
            Line:
                points: self.right, self.y, self.right, self.top
                width: 1

        # Logo block
        BoxLayout:
            size_hint_y: None
            height: 64
            orientation: 'vertical'
            spacing: 2
            padding: [0, 0, 0, 16]

            Label:
                text: 'SIGN·2·SIGN'
                font_name: 'Roboto'
                font_size: '16sp'
                bold: True
                color: 0, 0.9, 0.63, 1
                halign: 'left'
                text_size: self.size

            Label:
                text: 'Gesture Intelligence'
                font_size: '10sp'
                color: 1, 1, 1, 0.28
                halign: 'left'
                text_size: self.size

        Widget:
            size_hint_y: None
            height: 1
            canvas:
                Color:
                    rgba: 0, 0.9, 0.63, 0.12
                Line:
                    points: self.x, self.center_y, self.right, self.center_y
                    width: 0.8

        # Nav buttons
        SideNavBtn:
            text: '  🏠  HOME'
            on_press: root.show_section('home')

        SideNavBtn:
            text: '  📷  LIVE CAM'
            on_press: root.show_section('live')
            id: nav_live

        SideNavBtn:
            text: '  ➕  ADD GESTURE'
            on_press: root.show_section('add')

        SideNavBtn:
            text: '  📊  HISTORY'
            on_press: root.show_section('history')

        Widget:  # spacer

        # Status card
        BoxLayout:
            size_hint_y: None
            height: 90
            orientation: 'vertical'
            padding: [12, 10, 12, 10]
            spacing: 4
            canvas.before:
                Color:
                    rgba: 0, 0.9, 0.63, 0.04
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [12]
                Color:
                    rgba: 0, 0.9, 0.63, 0.1
                Line:
                    rounded_rectangle: self.x, self.y, self.width, self.height, 12
                    width: 0.8

            BoxLayout:
                size_hint_y: None
                height: 20
                Label:
                    text: 'MODEL STATUS'
                    font_size: '9sp'
                    color: 1, 1, 1, 0.3
                    halign: 'left'
                    text_size: self.size
                    bold: True
                    letter_spacing: 2

            Label:
                text: ('✅ LOADED' if root.model_ok else '❌ NOT FOUND')
                font_size: '12sp'
                color: (0, 0.9, 0.63, 1) if root.model_ok else (1, 0.4, 0.4, 1)
                halign: 'left'
                text_size: self.size
                bold: True

            Label:
                text: f'{len(root.gesture_labels)} gestures'
                font_size: '11sp'
                color: 1, 1, 1, 0.35
                halign: 'left'
                text_size: self.size

    # ── MAIN CONTENT AREA ─────────────────────────
    BoxLayout:
        id: content_area
        orientation: 'vertical'
        padding: [24, 20, 24, 20]
        spacing: 16

        # ═══ HOME SECTION ═══════════════════════════
        BoxLayout:
            id: section_home
            orientation: 'vertical'
            spacing: 16

            # Header
            BoxLayout:
                size_hint_y: None
                height: 70
                orientation: 'vertical'
                spacing: 4
                Label:
                    text: 'SPEAK WITH YOUR HANDS'
                    font_size: '26sp'
                    bold: True
                    color: 1, 1, 1, 0.92
                    halign: 'left'
                    text_size: self.size
                Label:
                    text: 'Real-time gesture recognition · Bridging silence into connection'
                    font_size: '13sp'
                    color: 1, 1, 1, 0.3
                    halign: 'left'
                    text_size: self.size

            # Stat row
            BoxLayout:
                size_hint_y: None
                height: 90
                spacing: 12

                StatCard:
                    icon_text: '✋'
                    stat_num: str(len(root.gesture_labels))
                    stat_label: 'GESTURES'

                StatCard:
                    icon_text: '⚡'
                    stat_num: '5'
                    stat_label: 'FPS PREDICT'

                StatCard:
                    icon_text: '🎯'
                    stat_num: '64px'
                    stat_label: 'ROI SIZE'

                StatCard:
                    icon_text: '📊'
                    stat_num: '100'
                    stat_label: 'SAMPLES'

            # How it works
            Label:
                size_hint_y: None
                height: 28
                text: 'HOW IT WORKS'
                font_size: '10sp'
                bold: True
                color: 0, 0.9, 0.63, 0.5
                halign: 'left'
                text_size: self.size

            BoxLayout:
                spacing: 12

                StepCard:
                    step_num: '01'
                    step_icon: '📷'
                    step_title: 'CAPTURE'
                    step_desc: 'Webcam streams frames. Center ROI box isolates your hand.'

                StepCard:
                    step_num: '02'
                    step_icon: '✂️'
                    step_title: 'EXTRACT'
                    step_desc: 'ROI resized to 64×64 and normalized for the CNN input.'

                StepCard:
                    step_num: '03'
                    step_icon: '🧠'
                    step_title: 'PREDICT'
                    step_desc: 'TensorFlow classifies gesture with a confidence score.'

                StepCard:
                    step_num: '04'
                    step_icon: '➕'
                    step_title: 'TRAIN'
                    step_desc: 'Collect 100 samples and retrain — no code needed.'

            # CTA buttons
            BoxLayout:
                size_hint_y: None
                height: 48
                spacing: 12

                Button:
                    text: '▶  LAUNCH CAMERA'
                    font_size: '13sp'
                    bold: True
                    background_color: 0, 0, 0, 0
                    color: 0.02, 0.05, 0.04, 1
                    on_press: root.show_section('live')
                    canvas.before:
                        Color:
                            rgba: 0, 0.9, 0.63, 1
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [12]

                Button:
                    text: '➕  ADD GESTURE'
                    font_size: '13sp'
                    bold: True
                    background_color: 0, 0, 0, 0
                    color: 0.96, 0.62, 0.04, 1
                    on_press: root.show_section('add')
                    canvas.before:
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.1
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [12]
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.28
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 12
                            width: 1

        # ═══ LIVE CAM SECTION ════════════════════════
        BoxLayout:
            id: section_live
            orientation: 'vertical'
            spacing: 12
            opacity: 0
            disabled: True

            # Top bar
            BoxLayout:
                size_hint_y: None
                height: 40
                Label:
                    text: 'LIVE RECOGNITION'
                    font_size: '20sp'
                    bold: True
                    color: 1, 1, 1, 0.9
                    halign: 'left'
                    text_size: self.size
                Label:
                    text: root.live_status_text
                    font_size: '11sp'
                    color: (0, 0.9, 0.63, 1) if root.predicting else (1,1,1,0.28)
                    halign: 'right'
                    text_size: self.size

            # Cam + predictions row
            BoxLayout:
                spacing: 14

                # Camera feed
                RelativeLayout:
                    size_hint_x: 0.58
                    canvas.before:
                        Color:
                            rgba: 0, 0, 0, 0.6
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [16]
                        Color:
                            rgba: 0, 0.9, 0.63, 0.14
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 16
                            width: 1

                    Camera:
                        id: camera
                        resolution: (640, 480)
                        play: True
                        allow_stretch: True
                        keep_ratio: True

                    # ROI overlay
                    Widget:
                        canvas:
                            Color:
                                rgba: 0, 0.9, 0.63, 0.35
                            Line:
                                rectangle: self.x + self.width*0.25, self.y + self.height*0.22, self.width*0.5, self.height*0.56
                                width: 2
                            # Corner accents
                            Color:
                                rgba: 0, 0.9, 0.63, 0.9
                            Line:
                                points: [self.x+self.width*0.25, self.y+self.height*0.22+16,
                                         self.x+self.width*0.25, self.y+self.height*0.22,
                                         self.x+self.width*0.25+16, self.y+self.height*0.22]
                                width: 2.5
                            Line:
                                points: [self.x+self.width*0.75-16, self.y+self.height*0.22,
                                         self.x+self.width*0.75, self.y+self.height*0.22,
                                         self.x+self.width*0.75, self.y+self.height*0.22+16]
                                width: 2.5
                            Line:
                                points: [self.x+self.width*0.25, self.y+self.height*0.78-16,
                                         self.x+self.width*0.25, self.y+self.height*0.78,
                                         self.x+self.width*0.25+16, self.y+self.height*0.78]
                                width: 2.5
                            Line:
                                points: [self.x+self.width*0.75-16, self.y+self.height*0.78,
                                         self.x+self.width*0.75, self.y+self.height*0.78,
                                         self.x+self.width*0.75, self.y+self.height*0.78-16]
                                width: 2.5

                    Label:
                        text: 'Place hand in frame'
                        pos_hint: {'center_x': 0.5, 'y': 0.02}
                        size_hint: 1, 0.08
                        font_size: '10sp'
                        color: 0, 0.9, 0.63, 0.4
                        bold: True

                # Right panel — predictions
                BoxLayout:
                    size_hint_x: 0.42
                    orientation: 'vertical'
                    spacing: 10

                    # Big prediction card
                    BoxLayout:
                        orientation: 'vertical'
                        spacing: 6
                        padding: [16, 14, 16, 14]
                        canvas.before:
                            Color:
                                rgba: 0, 0.9, 0.63, 0.05
                            RoundedRectangle:
                                pos: self.pos
                                size: self.size
                                radius: [14]
                            Color:
                                rgba: 0, 0.9, 0.63, 0.14
                            Line:
                                rounded_rectangle: self.x, self.y, self.width, self.height, 14
                                width: 1

                        Label:
                            text: 'DETECTED GESTURE'
                            font_size: '9sp'
                            color: 1, 1, 1, 0.3
                            halign: 'left'
                            text_size: self.size
                            bold: True
                            size_hint_y: None
                            height: 18

                        Label:
                            text: root.prediction_text
                            font_size: '32sp'
                            bold: True
                            color: root.prediction_color
                            halign: 'left'
                            text_size: self.size

                        Label:
                            text: 'Confidence: ' + root.confidence_text
                            font_size: '12sp'
                            color: 0.96, 0.62, 0.04, 0.8
                            halign: 'left'
                            text_size: self.size
                            size_hint_y: None
                            height: 22

                    # History card
                    BoxLayout:
                        orientation: 'vertical'
                        spacing: 4
                        padding: [14, 12, 14, 12]
                        canvas.before:
                            Color:
                                rgba: 0.96, 0.62, 0.04, 0.04
                            RoundedRectangle:
                                pos: self.pos
                                size: self.size
                                radius: [14]
                            Color:
                                rgba: 0.96, 0.62, 0.04, 0.12
                            Line:
                                rounded_rectangle: self.x, self.y, self.width, self.height, 14
                                width: 1

                        Label:
                            text: 'RECENT GESTURES'
                            font_size: '9sp'
                            color: 1, 1, 1, 0.3
                            halign: 'left'
                            text_size: self.size
                            bold: True
                            size_hint_y: None
                            height: 18

                        Label:
                            text: root.history_text
                            font_size: '12sp'
                            color: 1, 1, 1, 0.55
                            halign: 'left'
                            text_size: self.size

            # Control buttons
            BoxLayout:
                size_hint_y: None
                height: 44
                spacing: 12

                Button:
                    text: ('⏸  STOP PREDICTION' if root.predicting else '▶  START PREDICTION')
                    font_size: '13sp'
                    bold: True
                    background_color: 0, 0, 0, 0
                    color: (0.02, 0.05, 0.04, 1) if root.predicting else (0, 0.9, 0.63, 1)
                    on_press: root.toggle_prediction()
                    canvas.before:
                        Color:
                            rgba: (0, 0.9, 0.63, 1) if root.predicting else (0, 0.9, 0.63, 0.08)
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [10]
                        Color:
                            rgba: (0, 0, 0, 0) if root.predicting else (0, 0.9, 0.63, 0.25)
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 10
                            width: 1

                Button:
                    text: '➕  ADD NEW GESTURE'
                    font_size: '13sp'
                    bold: True
                    background_color: 0, 0, 0, 0
                    color: 0.96, 0.62, 0.04, 1
                    on_press: root.show_add_gesture_popup()
                    canvas.before:
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.08
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [10]
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.25
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 10
                            width: 1

        # ═══ ADD GESTURE SECTION ═════════════════════
        BoxLayout:
            id: section_add
            orientation: 'vertical'
            spacing: 14
            opacity: 0
            disabled: True

            Label:
                size_hint_y: None
                height: 40
                text: 'ADD NEW GESTURE'
                font_size: '20sp'
                bold: True
                color: 1, 1, 1, 0.9
                halign: 'left'
                text_size: self.size

            BoxLayout:
                spacing: 14

                # Input card
                BoxLayout:
                    orientation: 'vertical'
                    spacing: 12
                    padding: [20, 18, 20, 18]
                    canvas.before:
                        Color:
                            rgba: 0, 0.9, 0.63, 0.04
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [16]
                        Color:
                            rgba: 0, 0.9, 0.63, 0.12
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 16
                            width: 1

                    Label:
                        text: 'GESTURE NAME'
                        font_size: '9sp'
                        color: 1, 1, 1, 0.3
                        halign: 'left'
                        text_size: self.size
                        bold: True
                        size_hint_y: None
                        height: 18

                    TextInput:
                        id: add_gesture_name
                        hint_text: 'e.g. HELLO, YES, STOP...'
                        multiline: False
                        font_size: '16sp'
                        foreground_color: 1, 1, 1, 0.9
                        hint_text_color: 1, 1, 1, 0.2
                        background_color: 0.04, 0.1, 0.07, 1
                        cursor_color: 0, 0.9, 0.63, 1
                        padding: [12, 10, 12, 10]
                        size_hint_y: None
                        height: 44

                    Label:
                        text: '100 samples collected automatically over ~30 seconds using your webcam.'
                        font_size: '11sp'
                        color: 1, 1, 1, 0.3
                        halign: 'left'
                        text_size: self.size
                        size_hint_y: None
                        height: 40

                    Button:
                        text: '📸  START COLLECTION'
                        font_size: '13sp'
                        bold: True
                        size_hint_y: None
                        height: 44
                        background_color: 0, 0, 0, 0
                        color: 0.02, 0.05, 0.04, 1
                        on_press: root.start_add_gesture()
                        canvas.before:
                            Color:
                                rgba: 0, 0.9, 0.63, 1
                            RoundedRectangle:
                                pos: self.pos
                                size: self.size
                                radius: [10]

                    Label:
                        text: 'After collecting all gestures, run:\\npython improved_train_model.py'
                        font_size: '10sp'
                        color: 0.96, 0.62, 0.04, 0.6
                        halign: 'left'
                        text_size: self.size

                # Existing gestures list
                BoxLayout:
                    orientation: 'vertical'
                    spacing: 8
                    padding: [16, 14, 16, 14]
                    canvas.before:
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.04
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [16]
                        Color:
                            rgba: 0.96, 0.62, 0.04, 0.1
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 16
                            width: 1

                    Label:
                        text: 'TRAINED GESTURES'
                        font_size: '9sp'
                        color: 1, 1, 1, 0.3
                        halign: 'left'
                        text_size: self.size
                        bold: True
                        size_hint_y: None
                        height: 18

                    Label:
                        text: root.gesture_list_text
                        font_size: '13sp'
                        color: 0.96, 0.62, 0.04, 0.75
                        halign: 'left'
                        text_size: self.size
                        markup: True

        # ═══ HISTORY SECTION ═════════════════════════
        BoxLayout:
            id: section_history
            orientation: 'vertical'
            spacing: 12
            opacity: 0
            disabled: True

            Label:
                size_hint_y: None
                height: 40
                text: 'GESTURE HISTORY'
                font_size: '20sp'
                bold: True
                color: 1, 1, 1, 0.9
                halign: 'left'
                text_size: self.size

            BoxLayout:
                orientation: 'vertical'
                spacing: 8
                padding: [20, 16, 20, 16]
                canvas.before:
                    Color:
                        rgba: 0, 0.9, 0.63, 0.04
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [16]
                    Color:
                        rgba: 0, 0.9, 0.63, 0.11
                    Line:
                        rounded_rectangle: self.x, self.y, self.width, self.height, 16
                        width: 1

                Label:
                    text: 'RECENT PREDICTIONS'
                    font_size: '9sp'
                    color: 1, 1, 1, 0.3
                    halign: 'left'
                    text_size: self.size
                    bold: True
                    size_hint_y: None
                    height: 18

                Label:
                    id: history_full_label
                    text: root.history_text if root.history_text else 'No predictions yet. Start the camera and make gestures!'
                    font_size: '14sp'
                    color: 1, 1, 1, 0.55
                    halign: 'left'
                    text_size: self.size

                Button:
                    text: '🗑  CLEAR HISTORY'
                    size_hint_y: None
                    height: 36
                    font_size: '11sp'
                    bold: True
                    background_color: 0, 0, 0, 0
                    color: 1, 0.42, 0.42, 0.7
                    on_press: root.clear_history()
                    canvas.before:
                        Color:
                            rgba: 1, 0.42, 0.42, 0.06
                        RoundedRectangle:
                            pos: self.pos
                            size: self.size
                            radius: [8]
                        Color:
                            rgba: 1, 0.42, 0.42, 0.18
                        Line:
                            rounded_rectangle: self.x, self.y, self.width, self.height, 8
                            width: 0.8

<SideNavBtn@Button>:
    size_hint_y: None
    height: 40
    font_size: '12sp'
    halign: 'left'
    background_color: 0, 0, 0, 0
    background_normal: ''
    color: 1, 1, 1, 0.35
    padding: [12, 0, 0, 0]
    canvas.before:
        Color:
            rgba: 0, 0.9, 0.63, 0.07 if self.state == 'down' else 0
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [8]

<StatCard@BoxLayout>:
    icon_text: '✋'
    stat_num: '0'
    stat_label: 'STAT'
    orientation: 'vertical'
    spacing: 2
    padding: [12, 10, 12, 10]
    canvas.before:
        Color:
            rgba: 0, 0.9, 0.63, 0.04
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [12]
        Color:
            rgba: 0, 0.9, 0.63, 0.1
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, 12
            width: 0.8
    Label:
        text: root.icon_text
        font_size: '18sp'
        size_hint_y: None
        height: 28
    Label:
        text: root.stat_num
        font_size: '18sp'
        bold: True
        color: 0, 0.9, 0.63, 1
    Label:
        text: root.stat_label
        font_size: '8sp'
        color: 1, 1, 1, 0.28
        bold: True

<StepCard@BoxLayout>:
    step_num: '01'
    step_icon: '📷'
    step_title: 'STEP'
    step_desc: 'Description here.'
    orientation: 'vertical'
    spacing: 6
    padding: [14, 12, 14, 12]
    canvas.before:
        Color:
            rgba: 0, 0.9, 0.63, 0.03
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [12]
        Color:
            rgba: 0, 0.9, 0.63, 0.09
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, 12
            width: 0.8
    Label:
        text: root.step_icon
        font_size: '22sp'
        size_hint_y: None
        height: 32
    Label:
        text: root.step_title
        font_size: '10sp'
        bold: True
        color: 0, 0.9, 0.63, 0.85
        halign: 'left'
        text_size: self.size
        size_hint_y: None
        height: 18
    Label:
        text: root.step_desc
        font_size: '10sp'
        color: 1, 1, 1, 0.3
        halign: 'left'
        text_size: self.size

<AddGesturePopup>:
    size_hint: 0.52, 0.44
    title: ''
    separator_height: 0
    background: ''
    background_color: 0, 0, 0, 0

    BoxLayout:
        orientation: 'vertical'
        padding: [24, 20, 24, 20]
        spacing: 14
        canvas.before:
            Color:
                rgba: 0.04, 0.1, 0.07, 0.97
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [18]
            Color:
                rgba: 0, 0.9, 0.63, 0.18
            Line:
                rounded_rectangle: self.x, self.y, self.width, self.height, 18
                width: 1.2

        Label:
            text: 'ADD NEW GESTURE'
            font_size: '16sp'
            bold: True
            color: 0, 0.9, 0.63, 1
            size_hint_y: None
            height: 28
            halign: 'left'
            text_size: self.size

        Label:
            text: 'Enter a name (HELLO, YES, ROCK...)'
            font_size: '11sp'
            color: 1, 1, 1, 0.35
            size_hint_y: None
            height: 22
            halign: 'left'
            text_size: self.size

        TextInput:
            id: gesture_name
            multiline: False
            font_size: '18sp'
            foreground_color: 1, 1, 1, 0.9
            hint_text: 'Type gesture name...'
            hint_text_color: 1, 1, 1, 0.2
            background_color: 0.02, 0.06, 0.04, 1
            cursor_color: 0, 0.9, 0.63, 1
            padding: [14, 10, 14, 10]
            size_hint_y: None
            height: 48

        BoxLayout:
            size_hint_y: None
            height: 44
            spacing: 12

            Button:
                text: 'CANCEL'
                font_size: '12sp'
                bold: True
                background_color: 0, 0, 0, 0
                color: 1, 0.42, 0.42, 0.8
                on_press: root.dismiss()
                canvas.before:
                    Color:
                        rgba: 1, 0.42, 0.42, 0.08
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [10]
                    Color:
                        rgba: 1, 0.42, 0.42, 0.2
                    Line:
                        rounded_rectangle: self.x, self.y, self.width, self.height, 10
                        width: 1

            Button:
                text: '📸  START COLLECTION'
                font_size: '12sp'
                bold: True
                background_color: 0, 0, 0, 0
                color: 0.02, 0.05, 0.04, 1
                on_press: root.save_gesture()
                canvas.before:
                    Color:
                        rgba: 0, 0.9, 0.63, 1
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [10]

<TrainGesturePopup>:
    size_hint: 0.46, 0.38
    title: ''
    separator_height: 0
    background: ''
    background_color: 0, 0, 0, 0

    BoxLayout:
        orientation: 'vertical'
        padding: [24, 20, 24, 20]
        spacing: 14
        canvas.before:
            Color:
                rgba: 0.04, 0.1, 0.07, 0.97
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [18]
            Color:
                rgba: 0.96, 0.62, 0.04, 0.2
            Line:
                rounded_rectangle: self.x, self.y, self.width, self.height, 18
                width: 1.2

        Label:
            text: root.status_text
            font_size: '15sp'
            bold: True
            color: 0, 0.9, 0.63, 1
            size_hint_y: None
            height: 28

        ProgressBar:
            max: 100
            value: root.progress_value
            size_hint_y: None
            height: 8

        Label:
            text: '👁  Hold your gesture steady in the frame'
            font_size: '12sp'
            color: 0.96, 0.62, 0.04, 0.75

        Button:
            text: 'CANCEL'
            size_hint_y: None
            height: 40
            font_size: '12sp'
            bold: True
            background_color: 0, 0, 0, 0
            color: 1, 0.42, 0.42, 0.8
            on_press: root.cancel_collection()
            canvas.before:
                Color:
                    rgba: 1, 0.42, 0.42, 0.08
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10]
                Color:
                    rgba: 1, 0.42, 0.42, 0.2
                Line:
                    rounded_rectangle: self.x, self.y, self.width, self.height, 10
                    width: 1
'''

Builder.load_string(KV)


# ── POPUP CLASSES ─────────────────────────────────────────────
class AddGesturePopup(Popup):
    def save_gesture(self):
        gesture_name = self.ids.gesture_name.text.strip().upper()
        if not gesture_name:
            print("❌ Please enter a gesture name!")
            return
        if gesture_name not in labels:
            labels.append(gesture_name)
            np.save('label_map.npy', np.array(labels))
        self.dismiss()
        train_popup = TrainGesturePopup(gesture_name=gesture_name)
        train_popup.open()


class TrainGesturePopup(Popup):
    status_text  = StringProperty("Collecting...")
    progress_value = NumericProperty(0)

    def __init__(self, gesture_name, **kwargs):
        self.gesture_name     = gesture_name
        self.samples_collected = 0
        self.total_samples     = 100
        self.is_collecting     = False
        super().__init__(**kwargs)
        self.title             = f"Collecting: {gesture_name}"
        self.status_text       = f"{gesture_name}: 0/{self.total_samples}"
        self.progress_value    = 0

    def on_open(self):
        self.is_collecting = True
        Clock.schedule_interval(self.capture_sample, 0.3)

    def cancel_collection(self):
        self.is_collecting = False
        Clock.unschedule(self.capture_sample)
        self.dismiss()

    def capture_sample(self, dt):
        if not self.is_collecting or self.samples_collected >= self.total_samples:
            Clock.unschedule(self.capture_sample)
            if self.samples_collected >= self.total_samples:
                self.dismiss()
                self._show_done()
            return

        app = App.get_running_app()
        if not hasattr(app, 'root') or not hasattr(app.root.ids, 'camera'):
            return
        camera = app.root.ids.camera
        if camera and hasattr(camera, 'texture') and camera.texture:
            try:
                buf   = camera.texture.pixels
                frame = np.frombuffer(buf, np.uint8).reshape(
                    (camera.texture.height, camera.texture.width, 4))[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                h, w  = frame.shape[:2]
                roi_s = min(h, w) // 2
                x1    = (w - roi_s) // 2
                y1    = (h - roi_s) // 2
                roi   = cv2.resize(frame[y1:y1+roi_s, x1:x1+roi_s], (64, 64))

                save_path = f"gesture_data/{self.gesture_name}"
                os.makedirs(save_path, exist_ok=True)
                existing  = [f for f in os.listdir(save_path) if f.endswith('.npy')]
                np.save(f"{save_path}/{len(existing)}.npy", roi)

                self.samples_collected += 1
                self.status_text        = f"{self.gesture_name}: {self.samples_collected}/{self.total_samples}"
                self.progress_value     = (self.samples_collected / self.total_samples) * 100
            except Exception as e:
                print(f"❌ Capture error: {e}")

    def _show_done(self):
        print(f"\n✅ Collection complete: {self.gesture_name} ({self.samples_collected} samples)")
        print("👉 Run: python improved_train_model.py")


# ── MAIN LAYOUT ───────────────────────────────────────────────
class Sign2SignLayout(BoxLayout):
    prediction_text  = StringProperty("Show your gesture")
    confidence_text  = StringProperty("0%")
    prediction_color = ListProperty([1, 1, 1, 0.5])
    history_text     = StringProperty("")
    predicting       = BooleanProperty(False)
    model_ok         = BooleanProperty(model_loaded)
    live_status_text = StringProperty("● STANDBY")
    gesture_labels   = ListProperty(labels)
    gesture_list_text = StringProperty("")

    SECTIONS = ['home', 'live', 'add', 'history']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gesture_history = deque(maxlen=20)
        self._update_gesture_list()

    def _update_gesture_list(self):
        if labels:
            self.gesture_list_text = '\n'.join(
                f"[color=00e5a0]✦[/color]  {g}" for g in labels)
        else:
            self.gesture_list_text = "No gestures trained yet."

    def show_section(self, name):
        for sec in self.SECTIONS:
            widget = self.ids.get(f'section_{sec}')
            if widget:
                if sec == name:
                    widget.opacity = 1
                    widget.disabled = False
                else:
                    widget.opacity = 0
                    widget.disabled = True

    def toggle_prediction(self):
        if not model_loaded:
            self.prediction_text  = "Model not loaded!"
            self.confidence_text  = "0%"
            self.prediction_color = [1, 0.4, 0.4, 1]
            return

        self.predicting = not self.predicting
        if self.predicting:
            self.live_status_text = "● LIVE"
            Clock.schedule_interval(self.update_prediction, 1.0 / 5)
        else:
            self.live_status_text = "● STANDBY"
            Clock.unschedule(self.update_prediction)
            self.prediction_text  = "Stopped"
            self.confidence_text  = "0%"
            self.prediction_color = [1, 0.4, 0.4, 1]

    def update_prediction(self, dt):
        if not self.predicting:
            return
        camera = self.ids.camera
        if not (camera and camera.texture):
            return
        try:
            buf   = camera.texture.pixels
            frame = np.frombuffer(buf, np.uint8).reshape(
                (camera.texture.height, camera.texture.width, 4))[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            h, w  = frame.shape[:2]
            roi_s = min(h, w) // 2
            x1    = (w - roi_s) // 2
            y1    = (h - roi_s) // 2
            roi   = cv2.resize(frame[y1:y1+roi_s, x1:x1+roi_s], (64, 64)) / 255.0

            preds      = model.predict(np.expand_dims(roi, axis=0), verbose=0)[0]
            pred_class = np.argmax(preds)
            confidence = np.max(preds)

            gesture = labels[pred_class] if pred_class < len(labels) else "Unknown"

            if confidence > 0.4:
                self.prediction_text  = gesture
                self.confidence_text  = f"{confidence*100:.1f}%"
                self.prediction_color = [0, 0.9, 0.63, 1]
                entry = f"{gesture}  ({confidence*100:.1f}%)"
                self.gesture_history.appendleft(entry)
                self.history_text = "\n".join(list(self.gesture_history)[:10])
            else:
                self.prediction_text  = "Unknown"
                self.confidence_text  = f"{confidence*100:.1f}%"
                self.prediction_color = [0.96, 0.62, 0.04, 1]
        except Exception as e:
            print(f"❌ Prediction error: {e}")

    def show_add_gesture_popup(self):
        AddGesturePopup().open()

    def start_add_gesture(self):
        name = self.ids.add_gesture_name.text.strip().upper()
        if not name:
            return
        if name not in labels:
            labels.append(name)
            np.save('label_map.npy', np.array(labels))
            self.gesture_labels = labels[:]
            self._update_gesture_list()
        popup = TrainGesturePopup(gesture_name=name)
        popup.open()

    def clear_history(self):
        self.gesture_history.clear()
        self.history_text = ""


# ── APP ENTRY ─────────────────────────────────────────────────
class Sign2SignApp(App):
    def build(self):
        Window.clearcolor = (0.02, 0.05, 0.04, 1)
        layout = Sign2SignLayout()
        layout.show_section('home')
        return layout


if __name__ == "__main__":
    print("\n" + "="*60)
    print("✋  SIGN2SIGN — Gesture Recognition")
    print("="*60)
    print("1. Click 'ADD GESTURE' to collect training data")
    print("2. Collect 100 samples per gesture")
    print("3. Run: python improved_train_model.py")
    print("4. Click 'LAUNCH CAMERA' → 'START PREDICTION'")
    print("="*60 + "\n")
    Sign2SignApp().run()
