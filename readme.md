# Fingers Dashboard AI

A real-time Python computer vision project that detects a hand from the webcam, recognizes common hand gestures, estimates the number of raised fingers, and displays the result inside a small dashboard at the bottom of the camera window.

The project uses:

- **OpenCV (`cv2`)** for webcam access, drawing, and displaying the window.
- **MediaPipe Tasks** for hand detection, hand landmarks, and gesture recognition.
- A fallback geometry-based counter for gestures that are not directly recognized by the official gesture model.

---

## Features

- Real-time webcam hand tracking.
- Dashboard inside the same OpenCV window.
- Shows the number presented by the hand.
- Uses MediaPipe Gesture Recognizer for stable known gestures.
- Uses geometry fallback for unsupported finger-count gestures.
- Supports closing the app with:
  - `ESC`
  - `q`
  - the window close button `X`
- Suppresses most TensorFlow / MediaPipe terminal logs.
- Code is split into clean modules.

---

## Project Structure

```txt
fingers-dashboard-AI/
│
├── ai_model/
│   ├── hand_landmarker.task
│   └── gesture_recognizer.task
│
├── doc/
│   └── hand_landmarks.png
│
├── fingerWithDashboard/
│   ├── main.py
│   ├── config.py
│   ├── logs.py
│   ├── hand_detector.py
│   ├── finger_counter.py
│   ├── drawing.py
│   ├── dashboard.py
│   └── camera_app.py
│
├── simple_code_test/
│   └── test1.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create and activate a virtual environment

On Git Bash / Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

On PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Current `requirements.txt`:

```txt
opencv-python
mediapipe
```

---

## Download the AI Models

Create the model folder:

```bash
mkdir -p ai_model
```

Download the hand landmark model:

```bash
curl -L -o ai_model/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Download the gesture recognizer model:

```bash
curl -L -o ai_model/gesture_recognizer.task https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task
```

The dashboard version mainly uses:

```txt
gesture_recognizer.task
```

The `hand_landmarker.task` model is kept because it is useful for simple tests and landmark-only experiments.

---

## Run the Project

From the project root:

```bash
python fingerWithDashboard/main.py
```

Or from inside `fingerWithDashboard`:

```bash
python main.py
```

---

## How It Works

The pipeline is:

```txt
Webcam frame
   ↓
OpenCV reads the frame
   ↓
MediaPipe Gesture Recognizer processes the frame
   ↓
The model returns:
   - hand landmarks
   - world landmarks
   - gesture category
   ↓
finger_counter.py decides the number
   ↓
dashboard.py displays the result
```

---

## AI Models Used

### `hand_landmarker.task`

This model detects the hand and returns 21 landmarks.

Examples:

```txt
4  = thumb tip
8  = index finger tip
12 = middle finger tip
16 = ring finger tip
20 = pinky tip
```

It answers:

```txt
Where are the hand points?
```

It does not directly understand gestures like `Victory` or `Open_Palm`.

---

### `gesture_recognizer.task`

This model is stronger. It detects the hand and recognizes common gestures.

It can recognize labels such as:

```txt
Closed_Fist
Open_Palm
Pointing_Up
Victory
Thumb_Up
Thumb_Down
ILoveYou
```

The project maps these labels to numbers:

```python
GESTURE_TO_NUMBER = {
    "Closed_Fist": 0,
    "Pointing_Up": 1,
    "Victory": 2,
    "Open_Palm": 5,
    "Thumb_Up": 1,
    "ILoveYou": 3,
    "Thumb_Down": 1
}
```

---

## Finger Counting Logic

The main file responsible for deciding the number is:

```txt
fingerWithDashboard/finger_counter.py
```

The main function is:

```python
get_dashboard_data()
```

It works like this:

```txt
If MediaPipe recognizes a known gesture:
    use GESTURE_TO_NUMBER

Else:
    use geometry fallback
```

### Gesture model examples

```txt
Closed_Fist  → 0
Pointing_Up  → 1
Victory      → 2
Open_Palm    → 5
```

### Geometry fallback

The official gesture model does not directly provide labels like:

```txt
Three
Four
```

So the project uses landmark geometry to estimate which fingers are open.

The fallback checks things like:

- Is the finger straight?
- Is the fingertip far from the palm?
- Is the fingertip extended away from the base joint?
- Is the pinky really extended or just straight but folded?

---

## Dashboard

The dashboard is created in:

```txt
fingerWithDashboard/dashboard.py
```

It displays:

- Number shown
- Detected gesture
- Confidence score
- Detection source:
  - `gesture_model`
  - `geometry_fallback`

Example:

```txt
Number shown: 2
Gesture: Victory
Source: gesture_model
```

---

## Controls

```txt
ESC  → close the app
q    → close the app
X    → close the OpenCV window
```

---

## Important Files

### `main.py`

Starts the application.

### `logs.py`

Suppresses useless TensorFlow / MediaPipe logs.

### `config.py`

Stores settings like:

- model paths
- camera index
- confidence values
- dashboard height

### `hand_detector.py`

Loads the MediaPipe Gesture Recognizer model and runs detection on each frame.

### `finger_counter.py`

Decides what number the hand is showing.

### `drawing.py`

Draws hand landmarks and finger tips on the camera image.

### `dashboard.py`

Adds the bottom dashboard panel to the camera window.

### `camera_app.py`

Controls the webcam loop, window behavior, and app shutdown.

---

## Known Limitations

This project is good for prototyping, but it is not perfect.

Finger counting may still be wrong when:

- the hand is partially hidden,
- fingers overlap,
- the hand is too close or too far,
- lighting is bad,
- the hand is strongly rotated,
- the pinky is straight but folded against the palm,
- the gesture is not one of MediaPipe's built-in gesture classes.

For a highly accurate 0–5 classifier, the best future step is to train a custom gesture recognition model with explicit classes:

```txt
Zero
One
Two
Three
Four
Five
```

---

## Troubleshooting

### The camera does not open

Try changing this in `config.py`:

```python
CAMERA_INDEX = 0
```

to:

```python
CAMERA_INDEX = 1
```

or:

```python
CAMERA_INDEX = 2
```

---

### Model file not found

Make sure the models exist here:

```txt
ai_model/hand_landmarker.task
ai_model/gesture_recognizer.task
```

Download them again using the commands in the "Download the AI Models" section.

---

### The terminal shows many TensorFlow / MediaPipe logs

The project already tries to suppress logs in:

```txt
logs.py
```

Some native MediaPipe warnings may still appear sometimes. They are usually harmless.

---

### Finger count is unstable

Try:

- improving lighting,
- showing the full hand clearly,
- keeping the hand away from the image border,
- avoiding finger overlap,
- increasing `history_size` in `camera_app.py`.

Example:

```python
finger_counter = FingerCounter(history_size=7)
```

---

## Future Improvements

Possible improvements:

- Train a custom 0–5 gesture classifier.
- Add a graphical dashboard with charts.
- Add calibration per user.
- Add support for two-hand combinations.
- Add confidence bars for each finger.
- Export gesture events to another app.
- Use the detected number to control a UI dashboard.

---

## Author

Created by **Amr Slama** as a Python AI / computer vision experiment using OpenCV and MediaPipe.
