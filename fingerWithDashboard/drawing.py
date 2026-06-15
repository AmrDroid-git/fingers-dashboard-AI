import cv2

from config import FINGER_TIPS


def draw_hand_landmarks(frame, result):
    h, w, _ = frame.shape

    if not result.hand_landmarks:
        return frame

    for hand_landmarks in result.hand_landmarks:
        draw_all_landmarks(frame, hand_landmarks, w, h)
        draw_finger_tips(frame, hand_landmarks, w, h)

    return frame


def draw_all_landmarks(frame, hand_landmarks, frame_width, frame_height):
    for landmark in hand_landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def draw_finger_tips(frame, hand_landmarks, frame_width, frame_height):
    for tip_id in FINGER_TIPS:
        landmark = hand_landmarks[tip_id]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        cv2.putText(
            frame,
            str(tip_id),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )