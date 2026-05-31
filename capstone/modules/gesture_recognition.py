"""Gesture recognition module using MediaPipe face and hand landmarks."""

import os
import sys
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_GESTURE_MESSAGES, GESTURE_CONFIG


class GestureRecognizer:
    """Recognize nodding, head shaking, and hand raising from video frames."""

    def __init__(self):
        self.config = GESTURE_CONFIG
        self.messages = DEFAULT_GESTURE_MESSAGES
        self.motion_history = deque(maxlen=self.config["temporal_window"])
        self.last_trigger_time = 0.0
        self.last_gesture = None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def recognize(self, frame: np.ndarray) -> Dict:
        """Recognize a communication gesture from one frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        result = self._empty_result()
        nose_point = self._extract_nose_point(face_results)
        if nose_point is not None:
            self.motion_history.append((time.time(), nose_point[0], nose_point[1]))

        detected_gesture, confidence = self._detect_head_motion()
        if detected_gesture is None:
            detected_gesture, confidence = self._detect_hand_raise(hand_results, nose_point)

        if detected_gesture and self._can_trigger(detected_gesture):
            result = self._build_result(detected_gesture, confidence)
            self.last_gesture = detected_gesture
            self.last_trigger_time = time.time()

        return result

    def _empty_result(self) -> Dict:
        return {
            "gesture": None,
            "gesture_name": "Waiting...",
            "gesture_emoji": "",
            "message": "",
            "confidence": 0.0,
        }

    def _extract_nose_point(self, face_results) -> Optional[Tuple[float, float]]:
        """Return normalized nose-tip coordinates if a face is detected."""
        if not face_results.multi_face_landmarks:
            return None
        face_landmarks = face_results.multi_face_landmarks[0]
        nose_tip = face_landmarks.landmark[1]
        return float(nose_tip.x), float(nose_tip.y)

    def _detect_head_motion(self) -> Tuple[Optional[str], float]:
        """Detect nodding or head shaking from nose-tip landmark history."""
        if len(self.motion_history) < 6:
            return None, 0.0

        xs = np.array([item[1] for item in self.motion_history], dtype=float)
        ys = np.array([item[2] for item in self.motion_history], dtype=float)
        x_range = float(xs.max() - xs.min())
        y_range = float(ys.max() - ys.min())
        threshold = self.config["motion_threshold"]

        y_changes = np.diff(ys)
        x_changes = np.diff(xs)
        y_direction_changes = int(np.sum(np.sign(y_changes[1:]) != np.sign(y_changes[:-1])))
        x_direction_changes = int(np.sum(np.sign(x_changes[1:]) != np.sign(x_changes[:-1])))

        if y_range > threshold and y_range > x_range * 1.25 and y_direction_changes >= 1:
            confidence = min(0.95, 0.65 + y_range * 4)
            return "nodding", confidence

        if x_range > threshold and x_range > y_range * 1.25 and x_direction_changes >= 1:
            confidence = min(0.95, 0.65 + x_range * 4)
            return "head_shaking", confidence

        return None, 0.0

    def _detect_hand_raise(self, hand_results, nose_point: Optional[Tuple[float, float]]) -> Tuple[Optional[str], float]:
        """Detect hand raising by comparing hand landmarks with the face reference point."""
        if not hand_results.multi_hand_landmarks or nose_point is None:
            return None, 0.0

        nose_y = nose_point[1]
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            hand_reference_y = min(float(wrist.y), float(index_mcp.y), float(middle_mcp.y))
            if hand_reference_y < nose_y - 0.08:
                return "hand_raising", 0.88
        return None, 0.0

    def _can_trigger(self, gesture: str) -> bool:
        """Apply debounce control to prevent repeated triggers."""
        elapsed = time.time() - self.last_trigger_time
        return gesture != self.last_gesture or elapsed >= self.config["debounce_seconds"]

    def _build_result(self, gesture: str, confidence: float) -> Dict:
        message = self.messages.get(gesture, {"text": "Gesture detected", "display_name": gesture.title()})
        return {
            "gesture": gesture,
            "gesture_name": message.get("display_name", gesture.replace("_", " ").title()),
            "gesture_emoji": message.get("emoji", ""),
            "message": message.get("text", "Gesture detected"),
            "confidence": float(confidence),
        }

    def recognize_from_camera(self) -> None:
        """Run a simple camera demo for gesture recognition."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera.")
            return

        print("Gesture recognition demo. Press q to quit.")
        while True:
            success, frame = cap.read()
            if not success:
                break
            result = self.recognize(frame)
            if result["gesture"]:
                text = f"Gesture: {result['gesture_name']} ({result['confidence']:.0%})"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Message: {result['message']}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureRecognizer().recognize_from_camera()
