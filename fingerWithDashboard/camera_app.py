import cv2

from config import WINDOW_NAME, CAMERA_INDEX
from hand_detector import create_hand_landmarker, detect_hands
from drawing import draw_hand_landmarks
from finger_counter import FingerCounter
from dashboard import add_dashboard


def run_camera_app():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    finger_counter = FingerCounter(history_size=5)

    try:
        with create_hand_landmarker() as landmarker:
            while True:
                if is_window_closed():
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = detect_hands(landmarker, rgb_frame)

                dashboard_data = finger_counter.get_dashboard_data(result)

                frame = draw_hand_landmarks(frame, result)

                frame = add_dashboard(frame, dashboard_data)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q"):
                    break

                if is_window_closed():
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def is_window_closed():
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1