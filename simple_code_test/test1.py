import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass


# Used to generate timestamps for each video frame
import time

# OpenCV: used for camera access, image processing, drawing, and displaying windows
import cv2

# MediaPipe main package
import mediapipe as mp

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MediaPipe class shortcuts
# =========================

# BaseOptions is used to configure the model path and other base settings
BaseOptions = python.BaseOptions

# HandLandmarker is the AI task that detects hands and their landmarks
HandLandmarker = vision.HandLandmarker

# Options/settings for the HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

# Defines how MediaPipe receives data: IMAGE, VIDEO, or LIVE_STREAM
VisionRunningMode = vision.RunningMode


# =========================
# Window name
# =========================

# Name of the OpenCV window that will show the camera feed
WINDOW_NAME = "New MediaPipe HandLandmarker"


# =========================
# Hand detector configuration
# =========================

options = HandLandmarkerOptions(
    # Path to the MediaPipe hand landmarker model file
    base_options=BaseOptions(model_asset_path="../ai_model/hand_landmarker.task"),

    # VIDEO mode is used because we process webcam frames one by one
    running_mode=VisionRunningMode.VIDEO,

    # Maximum number of hands to detect at the same time
    num_hands=2,

    # Minimum confidence required to detect a hand
    min_hand_detection_confidence=0.5,

    # Minimum confidence required to confirm that a hand is present
    min_hand_presence_confidence=0.5,

    # Minimum confidence required to continue tracking the hand between frames
    min_tracking_confidence=0.5,
)


# =========================
# Open webcam
# =========================

# 0 means the default webcam of the PC
cap = cv2.VideoCapture(0)

# Create an OpenCV window that can be resized
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


try:
    # Create the MediaPipe hand detector using the options above
    with HandLandmarker.create_from_options(options) as landmarker:

        # Main loop: keeps reading frames from the camera
        while True:

            # If the user closes the window using X, stop the program
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Read one frame from the webcam
            ret, frame = cap.read()

            # If the frame was not read correctly, stop the program
            if not ret:
                break

            # Flip the frame horizontally to make it behave like a mirror
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR color format, but MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the RGB frame into a MediaPipe image object
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # MediaPipe VIDEO mode requires a timestamp for each frame
            timestamp_ms = int(time.time() * 1000)

            # Run hand detection on the current frame
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Get frame height and width
            # MediaPipe gives landmark coordinates between 0 and 1,
            # so we need width and height to convert them to pixels
            h, w, _ = frame.shape

            # Check if at least one hand was detected
            if result.hand_landmarks:

                # Loop over every detected hand
                for hand_landmarks in result.hand_landmarks:

                    # Draw all 21 hand landmarks
                    for landmark in hand_landmarks:

                        # Convert normalized landmark coordinates to real pixel coordinates
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Draw a small green circle on each landmark
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    # Finger tip landmark IDs in MediaPipe:
                    # 4  = thumb tip
                    # 8  = index finger tip
                    # 12 = middle finger tip
                    # 16 = ring finger tip
                    # 20 = pinky tip
                    finger_tips = [4, 8, 12, 16, 20]

                    # Highlight only the finger tips
                    for tip_id in finger_tips:

                        # Get the landmark of the current finger tip
                        lm = hand_landmarks[tip_id]

                        # Convert normalized coordinates to pixels
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        # Draw a bigger red circle on the finger tip
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                        # Write the landmark ID next to the finger tip
                        cv2.putText(
                            frame,
                            str(tip_id),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

            # Show the final frame with the detected landmarks
            cv2.imshow(WINDOW_NAME, frame)

            # Wait 1 millisecond for a keyboard key
            key = cv2.waitKey(1) & 0xFF

            # ESC or q closes the program
            if key == 27 or key == ord("q"):
                break

            # Check again if the window was closed using X
            # This is needed because OpenCV updates window events after waitKey()
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

finally:
    # Release the webcam
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()