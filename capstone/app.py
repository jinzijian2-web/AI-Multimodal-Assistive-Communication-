"""Main Flask application for the multimodal assistive communication system."""

import os
import sys
import threading
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SPEECH_CONFIG, TTS_CACHE_DIR, WEB_CONFIG
from modules.database import DatabaseManager
from modules.emotion_recognition import EmotionRecognizer
from modules.gesture_recognition import GestureRecognizer
from modules.speech_to_text import SpeechToText
from modules.text_to_speech import TextToSpeech

app = Flask(__name__)
CORS(app)

camera = None
camera_lock = threading.Lock()
emotion_recognizer = None
gesture_recognizer = None
speech_to_text = None
text_to_speech = None
db_manager = None

current_results: Dict = {
    "emotion": {"name": "Waiting...", "confidence": 0.0, "message": ""},
    "gesture": {"name": "Waiting...", "confidence": 0.0, "message": "", "emoji": ""},
    "speech": {"text": "", "confidence": 0.0},
}


def init_modules() -> None:
    """Initialize all core modules."""
    global emotion_recognizer, gesture_recognizer, speech_to_text, text_to_speech, db_manager

    print("=" * 60)
    print("Initializing Multimodal Assistive Communication System")
    print("=" * 60)

    db_manager = DatabaseManager()
    db_manager.init_database()

    print("1. Initializing facial expression recognition module...")
    emotion_recognizer = EmotionRecognizer()

    print("2. Initializing gesture recognition module...")
    gesture_recognizer = GestureRecognizer()

    print("3. Initializing speech-to-text module...")
    speech_to_text = SpeechToText(model_size=SPEECH_CONFIG["whisper_model"])

    print("4. Initializing text-to-speech module...")
    text_to_speech = TextToSpeech(language=SPEECH_CONFIG["tts_language"])

    print("=" * 60)
    print("All modules were initialized successfully.")
    print("=" * 60)


def get_camera():
    """Return a shared camera instance."""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Unable to open camera.")
            return None
    return camera


def choose_output_message(results: Dict) -> str:
    """Choose the current communication message to display and speak."""
    gesture = results.get("gesture", {})
    emotion = results.get("emotion", {})
    if gesture.get("confidence", 0) > 0 and gesture.get("message"):
        return gesture["message"]
    if emotion.get("confidence", 0) > 0 and emotion.get("message"):
        return emotion["message"]
    return ""


def draw_results(frame: np.ndarray, results: Dict) -> np.ndarray:
    """Draw English recognition results on the video frame."""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    emotion = results.get("emotion", {})
    gesture = results.get("gesture", {})
    output_message = choose_output_message(results)

    emotion_name = emotion.get("name", "Waiting...")
    emotion_conf = emotion.get("confidence", 0.0)
    gesture_name = gesture.get("name", "Waiting...")
    gesture_conf = gesture.get("confidence", 0.0)

    emotion_text = f"Emotion: {emotion_name} ({emotion_conf:.0%})" if emotion_conf > 0 else f"Emotion: {emotion_name}"
    gesture_text = f"Gesture: {gesture_name} ({gesture_conf:.0%})" if gesture_conf > 0 else f"Gesture: {gesture_name}"

    cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if output_message:
        cv2.putText(frame, f"Output: {output_message}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return frame


def update_recognition_results(frame: np.ndarray) -> None:
    """Run recognition modules on one frame and update the shared result state."""
    if emotion_recognizer is not None:
        emotion, confidence, emotion_msg = emotion_recognizer.predict(frame)
        current_results["emotion"] = {
            "name": emotion,
            "confidence": float(confidence),
            "message": emotion_msg.get("text", ""),
        }

    if gesture_recognizer is not None:
        gesture_result = gesture_recognizer.recognize(frame)
        if gesture_result.get("gesture"):
            current_results["gesture"] = {
                "name": gesture_result.get("gesture_name", "Waiting..."),
                "confidence": float(gesture_result.get("confidence", 0.0)),
                "message": gesture_result.get("message", ""),
                "emoji": gesture_result.get("gesture_emoji", ""),
            }
        else:
            current_results["gesture"] = {
                "name": "Waiting...",
                "confidence": 0.0,
                "message": "",
                "emoji": "",
            }


def generate_frames():
    """Generate MJPEG video frames for the web page."""
    cap = get_camera()
    if cap is None:
        return

    frame_count = 0
    while True:
        with camera_lock:
            success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            update_recognition_results(frame)

        frame = draw_results(frame, current_results)
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/")
def index():
    """Serve the main web interface."""
    return send_file("index.html")


@app.route("/video_feed")
def video_feed():
    """Return the live MJPEG video stream."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def get_status():
    """Return current recognition results."""
    return jsonify(current_results)


@app.route("/api/recognize_emotion", methods=["POST"])
def recognize_emotion_api():
    """Recognize facial expression from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image was provided."}), 400
    image_bytes = request.files["image"].read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    emotion, confidence, message = emotion_recognizer.predict(image)
    return jsonify({"emotion": emotion, "confidence": confidence, "message": message.get("text", "")})


@app.route("/api/recognize_gesture", methods=["POST"])
def recognize_gesture_api():
    """Recognize gesture from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image was provided."}), 400
    image_bytes = request.files["image"].read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400
    return jsonify(gesture_recognizer.recognize(image))


@app.route("/api/speech_to_text", methods=["POST"])
def speech_to_text_api():
    """Transcribe an uploaded audio clip."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file was provided."}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    filename = audio_file.filename or "recording.webm"
    suffix = Path(filename).suffix or ".webm"

    try:
        text = speech_to_text.transcribe_bytes(audio_bytes, suffix=suffix)
    except Exception as exc:
        return jsonify({"error": f"Speech recognition failed: {exc}"}), 500

    current_results["speech"] = {"text": text, "confidence": 0.0}
    return jsonify({"text": text})


@app.route("/api/text_to_speech", methods=["POST"])
def text_to_speech_api():
    """Generate an MP3 file from text and return a playable URL."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text was provided."}), 400

    try:
        audio_path = text_to_speech.synthesize(text)
    except Exception as exc:
        return jsonify({"error": f"Text-to-speech failed: {exc}"}), 500

    filename = Path(audio_path).name
    return jsonify({"status": "ok", "text": text, "audio_url": f"/api/audio/{filename}"})


@app.route("/api/audio/<path:filename>")
def get_audio(filename: str):
    """Serve a generated TTS audio file."""
    audio_path = Path(TTS_CACHE_DIR) / filename
    if not audio_path.exists():
        return jsonify({"error": "Audio file not found."}), 404
    return send_file(str(audio_path), mimetype="audio/mpeg")


@app.route("/api/mappings", methods=["GET"])
def get_mappings():
    """Return all custom gesture mappings."""
    return jsonify(db_manager.get_all_mappings())


@app.route("/api/mappings", methods=["POST"])
def add_mapping():
    """Add a custom gesture mapping."""
    data = request.get_json(silent=True) or {}
    gesture_name = (data.get("gesture_name") or "").strip()
    output_text = (data.get("output_text") or "").strip()
    display_name = (data.get("display_name") or gesture_name.replace("_", " ").title()).strip()
    if not gesture_name or not output_text:
        return jsonify({"error": "Both gesture_name and output_text are required."}), 400

    mapping_id = db_manager.add_mapping(
        gesture_name=gesture_name,
        display_name=display_name,
        output_text=output_text,
        output_speech=data.get("output_speech") or output_text,
    )
    return jsonify({"status": "ok", "id": mapping_id})


@app.route("/api/mappings/<int:mapping_id>", methods=["PUT"])
def update_mapping(mapping_id: int):
    """Update a custom gesture mapping."""
    data = request.get_json(silent=True) or {}
    db_manager.update_mapping(
        mapping_id,
        gesture_name=data["gesture_name"],
        display_name=data.get("display_name"),
        output_text=data["output_text"],
        output_speech=data.get("output_speech") or data["output_text"],
    )
    return jsonify({"status": "ok"})


@app.route("/api/mappings/<int:mapping_id>", methods=["DELETE"])
def delete_mapping(mapping_id: int):
    """Delete a custom gesture mapping."""
    db_manager.delete_mapping(mapping_id)
    return jsonify({"status": "ok"})


def main() -> None:
    """Application entry point."""
    init_modules()
    print(f"Starting web server at http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    app.run(
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        debug=WEB_CONFIG["debug"],
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
