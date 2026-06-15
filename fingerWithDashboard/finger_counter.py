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


def finger_straightness(landmarks, joint_ids):
    """
    Calculates how straight a finger is.

    The idea:
    - If the finger is open, the direct distance from base to tip is large.
    - If the finger is folded, the direct distance from base to tip becomes smaller.

    straightness = direct distance / full bone path length

    Result:
    - close to 1.0 means finger is straight/open
    - smaller value means finger is folded/closed
    """

    base = landmarks[joint_ids[0]]
    joint1 = landmarks[joint_ids[1]]
    joint2 = landmarks[joint_ids[2]]
    tip = landmarks[joint_ids[3]]

    direct_distance = distance(base, tip)

    full_finger_length = (
        distance(base, joint1) +
        distance(joint1, joint2) +
        distance(joint2, tip)
    )

    if full_finger_length == 0:
        return 0

    return direct_distance / full_finger_length


def project_on_palm_axis(point, wrist, palm_axis):
    """
    Projects a point onto the palm direction axis.

    Palm axis is usually the direction from wrist to middle finger base.

    This helps us know if a finger tip is really extending forward
    away from the palm.
    """

    return dot(vector(wrist, point), palm_axis)


def is_normal_finger_open(landmarks, mcp_id, pip_id, dip_id, tip_id):
    """
    Checks if index/middle/ring/pinky is open.

    This version is rotation-independent.

    It works even if the hand points:
    - up
    - down
    - left
    - right

    Because it checks finger straightness, not screen direction.
    """

    wrist = landmarks[0]

    mcp = landmarks[mcp_id]
    pip = landmarks[pip_id]
    dip = landmarks[dip_id]
    tip = landmarks[tip_id]

    straightness = finger_straightness(
        landmarks,
        [mcp_id, pip_id, dip_id, tip_id]
    )

    tip_farther_than_pip = distance(wrist, tip) > distance(wrist, pip) * 1.02
    tip_farther_than_mcp = distance(wrist, tip) > distance(wrist, mcp) * 1.20

    finger_is_straight = straightness > 0.72

    return finger_is_straight and tip_farther_than_pip and tip_farther_than_mcp

def is_thumb_open(landmarks):
    """
    Checks if the thumb is open.

    The thumb is different from the other fingers.
    It moves sideways, so we use:
    - thumb straightness
    - distance from palm center
    """

    wrist = landmarks[0]

    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    palm_center = average_points([
        wrist,
        index_mcp,
        middle_mcp,
        ring_mcp,
        pinky_mcp,
    ])

    palm_width = distance(index_mcp, pinky_mcp)

    straightness = finger_straightness(
        landmarks,
        [1, 2, 3, 4]
    )

    thumb_tip_from_palm = distance(thumb_tip, palm_center)
    thumb_ip_from_palm = distance(thumb_ip, palm_center)

    thumb_is_straight = straightness > 0.70
    thumb_far_from_palm = thumb_tip_from_palm > thumb_ip_from_palm * 1.08
    thumb_far_from_index = distance(thumb_tip, index_mcp) > palm_width * 0.60

    return thumb_is_straight and thumb_far_from_palm and thumb_far_from_index


def geometry_fallback_count(landmarks):
    """
    Counts fingers manually using rotation-independent geometry.

    This does not depend on whether the hand points up or down.
    It depends mostly on finger straightness.
    """

    thumb = is_thumb_open(landmarks)

    index = is_normal_finger_open(landmarks, 5, 6, 7, 8)
    middle = is_normal_finger_open(landmarks, 9, 10, 11, 12)
    ring = is_normal_finger_open(landmarks, 13, 14, 15, 16)
    pinky = is_normal_finger_open(landmarks, 17, 18, 19, 20)

    states = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

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