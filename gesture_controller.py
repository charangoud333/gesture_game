import mediapipe as mp
import cv2

class HandGestureDetector:
    def __init__(self, max_hands=1):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))
                self.mp_draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

        return landmarks