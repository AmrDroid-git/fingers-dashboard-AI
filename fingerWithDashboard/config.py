from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR.parent / "ai_model" / "hand_landmarker.task")
GESTURE_MODEL_PATH = str(BASE_DIR.parent / "ai_model" / "gesture_recognizer.task")

WINDOW_NAME = "New MediaPipe HandLandmarker"

CAMERA_INDEX = 0

MAX_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

FINGER_TIPS = [4, 8, 12, 16, 20]

DASHBOARD_HEIGHT = 120