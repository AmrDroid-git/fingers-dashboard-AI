from collections import Counter, deque
import math


# This dictionary maps known gestures from MediaPipe GestureRecognizer
# to the number that should appear in the dashboard.
#
# Example:
# "Victory" means ✌️, so we map it to 2.
# "Open_Palm" means full open hand, so we map it to 5.
GESTURE_TO_NUMBER = {
    "Closed_Fist": 0,
    "Pointing_Up": 1,
    "Victory": 2,
    "Open_Palm": 5,
    "Thumb_Up": 1,
    "ILoveYou": 3,
    "Thumb_Down": 1
}


def distance(point1, point2):
    """
    Calculates the 3D distance between two hand landmarks.

    Each MediaPipe landmark has:
    - x position
    - y position
    - z depth

    We use this to know if a finger tip is far from the palm,
    far from another finger, etc.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def dot(vector1, vector2):
    """
    Calculates the dot product between two 3D vectors.

    Dot product is used to project a point onto the palm direction.
    This helps us know if a finger is extended away from the wrist.
    """

    return (
        vector1[0] * vector2[0] +
        vector1[1] * vector2[1] +
        vector1[2] * vector2[2]
    )


def vector(point1, point2):
    """
    Creates a 3D vector going from point1 to point2.

    Example:
    vector(wrist, middle_mcp)
    gives the direction from the wrist to the middle finger base.
    """

    return (
        point2.x - point1.x,
        point2.y - point1.y,
        point2.z - point1.z,
    )


def normalize(vector_value):
    """
    Converts a vector to a unit vector.

    A unit vector keeps the same direction,
    but its length becomes 1.

    This is useful because we only care about direction,
    not the original vector size.
    """

    length = math.sqrt(
        vector_value[0] ** 2 +
        vector_value[1] ** 2 +
        vector_value[2] ** 2
    )

    # Avoid division by zero
    if length == 0:
        return (0, 0, 0)

    return (
        vector_value[0] / length,
        vector_value[1] / length,
        vector_value[2] / length,
    )


def get_best_landmarks(result, hand_index):
    """
    Gets the best available landmarks for one hand.

    MediaPipe can return:
    1. hand_world_landmarks:
       3D real-world-like coordinates, better for geometry.

    2. hand_landmarks:
       normal image coordinates.

    We prefer hand_world_landmarks because they are better
    for detecting if fingers are open or closed.
    """

    world_landmarks = getattr(result, "hand_world_landmarks", None)

    if world_landmarks and len(world_landmarks) > hand_index:
        return world_landmarks[hand_index]

    return result.hand_landmarks[hand_index]


def get_gesture(result, hand_index):
    """
    Gets the gesture name predicted by MediaPipe GestureRecognizer.

    Example returned gesture names:
    - Open_Palm
    - Victory
    - Closed_Fist
    - Pointing_Up
    - Thumb_Up
    - ILoveYou

    It returns:
    - gesture name
    - confidence score

    Example:
    ("Victory", 0.87)
    """

    gestures = getattr(result, "gestures", None)

    # No gesture data exists
    if not gestures:
        return None, 0

    # The requested hand index does not exist
    if hand_index >= len(gestures):
        return None, 0

    # No gesture detected for this hand
    if not gestures[hand_index]:
        return None, 0

    # Take the best gesture prediction
    best_gesture = gestures[hand_index][0]

    return best_gesture.category_name, best_gesture.score


def average_points(points):
    """
    Calculates the average point between multiple landmarks.

    We use this to estimate the palm center.

    Example:
    palm center = average of wrist + finger base landmarks.
    """

    count = len(points)

    class Point:
        pass

    point = Point()

    point.x = sum(p.x for p in points) / count
    point.y = sum(p.y for p in points) / count
    point.z = sum(p.z for p in points) / count

    return point


def project_on_palm_axis(point, wrist, palm_axis):
    """
    Projects a point onto the palm direction axis.

    Palm axis is usually the direction from wrist to middle finger base.

    This helps us know if a finger tip is really extending forward
    away from the palm.
    """

    return dot(vector(wrist, point), palm_axis)


def is_normal_finger_open(landmarks, mcp_id, pip_id, tip_id):
    """
    Checks if a normal finger is open.

    Normal fingers are:
    - index
    - middle
    - ring
    - pinky

    It does NOT handle the thumb because the thumb moves differently.

    Parameters:
    - mcp_id: base joint of the finger
    - pip_id: middle joint of the finger
    - tip_id: tip of the finger

    Example for index finger:
    MCP = 5
    PIP = 6
    TIP = 8
    """

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Palm axis points from the wrist toward the fingers
    palm_axis = normalize(vector(wrist, middle_mcp))

    # Palm length is used to calculate a dynamic margin
    palm_length = distance(wrist, middle_mcp)

    mcp = landmarks[mcp_id]
    pip = landmarks[pip_id]
    tip = landmarks[tip_id]

    # Project finger joints onto the palm axis
    mcp_projection = project_on_palm_axis(mcp, wrist, palm_axis)
    pip_projection = project_on_palm_axis(pip, wrist, palm_axis)
    tip_projection = project_on_palm_axis(tip, wrist, palm_axis)

    # Margin avoids counting small movements as open fingers
    margin = palm_length * 0.20

    # The finger tip should be clearly farther than the PIP joint
    tip_is_farther_than_pip = tip_projection > pip_projection + margin

    # The PIP joint should be farther than the MCP joint
    pip_is_farther_than_mcp = pip_projection > mcp_projection

    # The tip should also be physically farther from the wrist than the PIP joint
    tip_distance_is_larger = distance(wrist, tip) > distance(wrist, pip) * 1.05

    return tip_is_farther_than_pip and pip_is_farther_than_mcp and tip_distance_is_larger


def is_thumb_open(landmarks):
    """
    Checks if the thumb is open.

    The thumb is special because it moves sideways,
    not vertically like the other fingers.

    So we do not use the same logic as index/middle/ring/pinky.
    Instead, we check:
    - Is the thumb tip far from the palm center?
    - Is the thumb tip far from the index finger base?
    """

    wrist = landmarks[0]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Estimate palm center using wrist and finger bases
    palm_center = average_points([
        wrist,
        index_mcp,
        middle_mcp,
        ring_mcp,
        pinky_mcp,
    ])

    # Estimate palm width using index base and pinky base
    palm_width = distance(index_mcp, pinky_mcp)

    # Compare thumb tip distance from palm center
    thumb_tip_from_palm = distance(thumb_tip, palm_center)
    thumb_ip_from_palm = distance(thumb_ip, palm_center)

    # Thumb is probably open if the tip is farther than the IP joint
    thumb_far_from_palm = thumb_tip_from_palm > thumb_ip_from_palm * 1.15

    # Thumb is probably open if it is far enough from index finger base
    thumb_far_from_index = distance(thumb_tip, index_mcp) > palm_width * 0.75

    return thumb_far_from_palm and thumb_far_from_index


def geometry_fallback_count(landmarks):
    """
    Counts fingers manually using landmark geometry.

    This is used only when MediaPipe GestureRecognizer
    does not recognize a known gesture.

    Example:
    If the model does not say Open_Palm or Victory,
    we manually check which fingers are open.
    """

    thumb = is_thumb_open(landmarks)

    index = is_normal_finger_open(landmarks, 5, 6, 8)
    middle = is_normal_finger_open(landmarks, 9, 10, 12)
    ring = is_normal_finger_open(landmarks, 13, 14, 16)
    pinky = is_normal_finger_open(landmarks, 17, 18, 20)

    # Store each finger state
    states = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

    # True counts as 1, False counts as 0
    return sum(states.values()), states


class FingerCounter:
    """
    Main class responsible for deciding the number shown by the hand.

    It uses two methods:

    1. Gesture model:
       If MediaPipe recognizes a known gesture,
       we directly map it to a number.

    2. Geometry fallback:
       If the gesture model does not recognize the gesture,
       we manually count open fingers using landmarks.

    It also smooths the result using recent frame history,
    to avoid flickering numbers.
    """

    def __init__(self, history_size=5):
        """
        Initializes the finger counter.

        history_size controls how many previous results are remembered.

        Example:
        history_size = 5 means we keep the last 5 detected numbers
        and choose the most common one.
        """

        self.history = deque(maxlen=history_size)

    def get_dashboard_data(self, result):
        """
        Main function called by camera_app.py.

        It receives the MediaPipe result,
        decides the number shown,
        and returns data for the dashboard.

        Returned data includes:
        - number of detected hands
        - count for each hand
        - total number
        - gesture name
        - gesture confidence
        - source used: gesture_model or geometry_fallback
        """

        # If no hand is detected, reset everything
        if not result.hand_landmarks:
            self.history.clear()

            return {
                "hands_count": 0,
                "counts": [],
                "raw_total": 0,
                "total": 0,
                "gesture": "None",
                "gesture_score": 0,
                "source": "none",
                "message": "No hand detected",
            }

        counts = []
        gestures = []
        sources = []

        # Loop over every detected hand
        for hand_index in range(len(result.hand_landmarks)):

            # Try to get the official gesture prediction
            gesture_name, gesture_score = get_gesture(result, hand_index)

            # If the gesture is known and confident enough,
            # use the official gesture model result
            if gesture_name in GESTURE_TO_NUMBER and gesture_score >= 0.55:
                count = GESTURE_TO_NUMBER[gesture_name]
                source = "gesture_model"

            # Otherwise, count fingers manually using landmarks
            else:
                landmarks = get_best_landmarks(result, hand_index)
                count, _ = geometry_fallback_count(landmarks)
                source = "geometry_fallback"

            counts.append(count)
            gestures.append((gesture_name, gesture_score))
            sources.append(source)

        # If multiple hands are detected, add their numbers
        raw_total = sum(counts)

        # Save the current raw result in history
        self.history.append(raw_total)

        # Choose the most common result from recent frames
        # This avoids unstable flickering
        smoothed_total = Counter(self.history).most_common(1)[0][0]

        # For dashboard display, show the first hand's gesture
        best_gesture_name = gestures[0][0] if gestures else "None"
        best_gesture_score = gestures[0][1] if gestures else 0

        return {
            "hands_count": len(counts),
            "counts": counts,
            "raw_total": raw_total,
            "total": smoothed_total,
            "gesture": best_gesture_name,
            "gesture_score": best_gesture_score,
            "source": sources[0] if sources else "none",
            "message": "Hand detected",
        }