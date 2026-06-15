from collections import Counter, deque
import math


GESTURE_TO_NUMBER = {
    "Closed_Fist": 0,
    "Pointing_Up": 1,
    "Victory": 2,
    "Open_Palm": 5,
}


def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def dot(vector1, vector2):
    return (
        vector1[0] * vector2[0] +
        vector1[1] * vector2[1] +
        vector1[2] * vector2[2]
    )


def vector(point1, point2):
    return (
        point2.x - point1.x,
        point2.y - point1.y,
        point2.z - point1.z,
    )


def normalize(vector_value):
    length = math.sqrt(
        vector_value[0] ** 2 +
        vector_value[1] ** 2 +
        vector_value[2] ** 2
    )

    if length == 0:
        return (0, 0, 0)

    return (
        vector_value[0] / length,
        vector_value[1] / length,
        vector_value[2] / length,
    )


def get_best_landmarks(result, hand_index):
    world_landmarks = getattr(result, "hand_world_landmarks", None)

    if world_landmarks and len(world_landmarks) > hand_index:
        return world_landmarks[hand_index]

    return result.hand_landmarks[hand_index]


def get_gesture(result, hand_index):
    gestures = getattr(result, "gestures", None)

    if not gestures:
        return None, 0

    if hand_index >= len(gestures):
        return None, 0

    if not gestures[hand_index]:
        return None, 0

    best_gesture = gestures[hand_index][0]

    return best_gesture.category_name, best_gesture.score


def average_points(points):
    count = len(points)

    class Point:
        pass

    point = Point()

    point.x = sum(p.x for p in points) / count
    point.y = sum(p.y for p in points) / count
    point.z = sum(p.z for p in points) / count

    return point


def project_on_palm_axis(point, wrist, palm_axis):
    return dot(vector(wrist, point), palm_axis)


def is_normal_finger_open(landmarks, mcp_id, pip_id, tip_id):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    palm_axis = normalize(vector(wrist, middle_mcp))
    palm_length = distance(wrist, middle_mcp)

    mcp = landmarks[mcp_id]
    pip = landmarks[pip_id]
    tip = landmarks[tip_id]

    mcp_projection = project_on_palm_axis(mcp, wrist, palm_axis)
    pip_projection = project_on_palm_axis(pip, wrist, palm_axis)
    tip_projection = project_on_palm_axis(tip, wrist, palm_axis)

    margin = palm_length * 0.20

    tip_is_farther_than_pip = tip_projection > pip_projection + margin
    pip_is_farther_than_mcp = pip_projection > mcp_projection

    tip_distance_is_larger = distance(wrist, tip) > distance(wrist, pip) * 1.05

    return tip_is_farther_than_pip and pip_is_farther_than_mcp and tip_distance_is_larger


def is_thumb_open(landmarks):
    wrist = landmarks[0]
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

    thumb_tip_from_palm = distance(thumb_tip, palm_center)
    thumb_ip_from_palm = distance(thumb_ip, palm_center)

    thumb_far_from_palm = thumb_tip_from_palm > thumb_ip_from_palm * 1.15
    thumb_far_from_index = distance(thumb_tip, index_mcp) > palm_width * 0.75

    return thumb_far_from_palm and thumb_far_from_index


def geometry_fallback_count(landmarks):
    thumb = is_thumb_open(landmarks)

    index = is_normal_finger_open(landmarks, 5, 6, 8)
    middle = is_normal_finger_open(landmarks, 9, 10, 12)
    ring = is_normal_finger_open(landmarks, 13, 14, 16)
    pinky = is_normal_finger_open(landmarks, 17, 18, 20)

    states = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

    return sum(states.values()), states


class FingerCounter:
    def __init__(self, history_size=5):
        self.history = deque(maxlen=history_size)

    def get_dashboard_data(self, result):
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

        for hand_index in range(len(result.hand_landmarks)):
            gesture_name, gesture_score = get_gesture(result, hand_index)

            if gesture_name in GESTURE_TO_NUMBER and gesture_score >= 0.55:
                count = GESTURE_TO_NUMBER[gesture_name]
                source = "gesture_model"

            else:
                landmarks = get_best_landmarks(result, hand_index)
                count, _ = geometry_fallback_count(landmarks)
                source = "geometry_fallback"

            counts.append(count)
            gestures.append((gesture_name, gesture_score))
            sources.append(source)

        raw_total = sum(counts)

        self.history.append(raw_total)

        smoothed_total = Counter(self.history).most_common(1)[0][0]

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