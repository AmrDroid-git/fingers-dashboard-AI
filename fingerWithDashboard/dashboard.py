import cv2
import numpy as np

from config import DASHBOARD_HEIGHT


def add_dashboard(frame, dashboard_data):
    frame_height, frame_width, _ = frame.shape

    dashboard = np.zeros((DASHBOARD_HEIGHT, frame_width, 3), dtype=np.uint8)

    dashboard[:] = (20, 20, 20)

    hands_count = dashboard_data["hands_count"]
    counts = dashboard_data["counts"]
    total = dashboard_data["total"]
    raw_total = dashboard_data["raw_total"]
    gesture = dashboard_data["gesture"]
    gesture_score = dashboard_data["gesture_score"]
    source = dashboard_data["source"]

    cv2.putText(
        dashboard,
        "FINGERS DASHBOARD",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if hands_count == 0:
        main_text = "No hand detected"
        detail_text = "Show your hand clearly to the camera"

    elif hands_count == 1:
        main_text = f"Number shown: {total}"
        detail_text = f"Gesture: {gesture} ({gesture_score:.2f}) | Source: {source}"

    else:
        main_text = f"Total number shown: {total}"
        detail_text = f"Hands: {hands_count} | Raw: {counts} | Source: {source}"

    cv2.putText(
        dashboard,
        main_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        dashboard,
        detail_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        2
    )

    return cv2.vconcat([frame, dashboard])