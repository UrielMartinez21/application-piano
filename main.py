import sys
import cv2
import mediapipe as mp
import pygame
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

class HandPianoApp(QWidget):
    def __init__(self):
        super().__init__()

        # 🪟 Window setup
        self.setWindowTitle("🎹 Hand Piano")
        self.setFixedSize(1000, 720)
        # self.setWindowIcon(QIcon("icon.png"))

        # 🎨 Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI';
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QFrame#VideoFrame {
                border: 3px solid #007ACC;
                border-radius: 10px;
                background-color: #000;
            }
        """)

        # 🔤 Title
        self.title_label = QLabel("🎹 Hand Piano with Computer Vision")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 🎥 Video display area
        self.video_label = QLabel()
        self.video_label.setFixedSize(960, 540)

        self.video_frame = QFrame()
        self.video_frame.setObjectName("VideoFrame")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(5, 5, 5, 5)
        video_layout.addWidget(self.video_label)
        self.video_frame.setLayout(video_layout)

        # 🎮 Button
        self.toggle_button = QPushButton("Start Camera")
        self.toggle_button.clicked.connect(self.toggle_camera)

        # 📐 Layout setup
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_frame, alignment=Qt.AlignCenter)
        layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        # 🔊 Sounds & MediaPipe init
        pygame.mixer.init()
        path = os.path.join(os.getcwd(), 'sounds')
        self.sounds = {
            "Left": {f: pygame.mixer.Sound(f"{path}/left_{f}.wav") for f in ["thumb", "index", "middle", "ring", "pinky"]},
            "Right": {f: pygame.mixer.Sound(f"{path}/right_{f}.wav") for f in ["thumb", "index", "middle", "ring", "pinky"]}
        }

        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.finger_states = {h: {f: False for f in ["thumb", "index", "middle", "ring", "pinky"]} for h in ["Left", "Right"]}
        self.finger_tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        self.finger_pips = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.toggle_button.setText("Stop camera")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.clear()
            self.toggle_button.setText("Start camera")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                states = {}

                # Thumb
                if label == "Right":
                    states["thumb"] = landmarks[self.finger_tips["thumb"]].x >= landmarks[self.finger_pips["thumb"]].x
                else:
                    states["thumb"] = landmarks[self.finger_tips["thumb"]].x <= landmarks[self.finger_pips["thumb"]].x

                # Other fingers
                for f in ["index", "middle", "ring", "pinky"]:
                    states[f] = landmarks[self.finger_tips[f]].y > landmarks[self.finger_pips[f]].y

                # Play sounds on press
                for f in states:
                    if not self.finger_states[label][f] and states[f]:
                        print(f"{label} {f} presionado")
                        self.sounds[label][f].play()
                    self.finger_states[label][f] = states[f]

        # Convert image for Qt
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandPianoApp()
    window.show()
    sys.exit(app.exec_())
