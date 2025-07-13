import cv2
import time
from pynput.keyboard import Controller, Key
import mediapipe as mp

keyboard = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_locked = None
gesture_lock_time = 0
gesture_lock_cooldown = 1.0  


def get_gesture(landmarks, frame):
    if not landmarks or len(landmarks) < 21:
        return None

    y = lambda i: landmarks[i][2]
    x = lambda i: landmarks[i][1]

    def distance(i, j):
        return ((x(i) - x(j)) ** 2 + (y(i) - y(j)) ** 2) ** 0.5

    index_up = y(8) < y(6)
    middle_up = y(12) < y(10)
    ring_up = y(16) < y(14)
    pinky_up = y(20) < y(18)

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "roll"

    if distance(4, 8) < 30 and (middle_up or ring_up or pinky_up):
        return "jump"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "left"

    if index_up and middle_up and ring_up and not pinky_up:
        return "right"

    return None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))

    gesture = get_gesture(landmarks, frame)
    current_time = time.time()

    if gesture != gesture_locked:
        if gesture in ["left", "right", "jump", "roll"]:
            if current_time - gesture_lock_time > gesture_lock_cooldown:
                gesture_locked = gesture
                gesture_lock_time = current_time

                if gesture == "left":
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                elif gesture == "right":
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                elif gesture == "jump":
                    keyboard.press(Key.up)
                    keyboard.release(Key.up)
                elif gesture == "roll":
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
    else:
        pass

    if gesture is None:
        gesture_locked = None

    cv2.putText(frame, f"Gesture: {gesture if gesture else 'None'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Controlled Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
