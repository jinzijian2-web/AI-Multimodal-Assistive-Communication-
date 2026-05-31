"""Facial expression recognition module based on a CNN model trained on FER2013."""

import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_EMOTION_MESSAGES, EMOTION_CONFIG


tf.get_logger().setLevel("ERROR")


class EmotionRecognizer:
    """Detect faces and classify facial expressions."""

    def __init__(self, model_path: Optional[str] = None):
        self.image_size = EMOTION_CONFIG["image_size"]
        self.emotions = EMOTION_CONFIG["emotions"]
        self.display_names = EMOTION_CONFIG["display_names"]
        self.confidence_threshold = EMOTION_CONFIG["confidence_threshold"]
        self.messages = DEFAULT_EMOTION_MESSAGES
        self.model = self._load_model(model_path or EMOTION_CONFIG["model_path"])

        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        if self.face_cascade.empty():
            print("Face detector failed to load.")
        else:
            print("Face detector loaded successfully.")

    def _load_model(self, model_path: str):
        """Load the Keras model if the model file exists."""
        if os.path.exists(model_path):
            print(f"Loading emotion recognition model: {model_path}")
            return tf.keras.models.load_model(model_path, compile=False)
        print(f"Model file not found: {model_path}")
        print("The emotion module will run in fallback mode.")
        return None

    def detect_faces(self, image: np.ndarray):
        """Detect face regions from a BGR image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60),
        )
        return faces, gray

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Resize and normalize the detected face image."""
        resized = cv2.resize(face_image, self.image_size)
        normalized = resized.astype("float32") / 255.0
        return normalized.reshape(1, self.image_size[0], self.image_size[1], 1)

    def predict(self, image: np.ndarray) -> Tuple[str, float, Dict[str, str]]:
        """Predict the expression in one frame."""
        faces, gray = self.detect_faces(image)
        if len(faces) == 0:
            return "No face detected", 0.0, self.messages["no_face"]

        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        if self.model is not None:
            input_tensor = self.preprocess_face(face_roi)
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])
            emotion_key = self.emotions[predicted_index]
        else:
            emotion_key, confidence = self._fallback_predict(face_roi)

        if confidence < self.confidence_threshold:
            return "Uncertain", confidence, self.messages["uncertain"]

        display_name = self.display_names.get(emotion_key, emotion_key.title())
        message = self.messages.get(emotion_key, self.messages["neutral"])
        return display_name, confidence, message

    def _fallback_predict(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Simple fallback prediction used when no trained model is available."""
        brightness = float(np.mean(face_image))
        if brightness > 150:
            return "happy", 0.70
        if brightness < 80:
            return "sad", 0.60
        return "neutral", 0.55

    def predict_from_file(self, image_path: str) -> Dict:
        """Predict expression from an image file."""
        image = cv2.imread(image_path)
        if image is None:
            return {"emotion": None, "confidence": 0.0, "message": {"text": "Unable to read image"}}
        emotion, confidence, message = self.predict(image)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "message": message,
            "image_path": image_path,
        }


if __name__ == "__main__":
    recognizer = EmotionRecognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera.")
        raise SystemExit(1)

    print("Press q to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            break
        emotion, confidence, message = recognizer.predict(frame)
        cv2.putText(frame, f"Emotion: {emotion} ({confidence:.0%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if message.get("text"):
            cv2.putText(frame, f"Message: {message['text']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Emotion Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
