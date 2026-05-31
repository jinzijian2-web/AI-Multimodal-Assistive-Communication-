"""Project configuration for the multimodal assistive communication system."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
TTS_CACHE_DIR = BASE_DIR / "tts_cache"
DATABASE_DIR = BASE_DIR / "database"

for directory in [DATA_DIR, MODELS_DIR, LOG_DIR, TTS_CACHE_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

EMOTION_CONFIG = {
    "model_path": str(MODELS_DIR / "emotion_model.h5"),
    "image_size": (48, 48),
    "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "display_names": {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "surprise": "Surprise",
        "uncertain": "Uncertain",
        "no_face": "No face detected",
    },
    "confidence_threshold": 0.50,
}

GESTURE_CONFIG = {
    "confidence_threshold": 0.60,
    "temporal_window": 12,
    "motion_threshold": 0.035,
    "debounce_seconds": 1.5,
    "gestures": ["nodding", "head_shaking", "hand_raising"],
}

SPEECH_CONFIG = {
    "whisper_model": "base",
    "speech_language": "en",
    "tts_language": "en",
    "tts_slow": False,
}

DATABASE_CONFIG = {
    "path": str(DATABASE_DIR / "mappings.db"),
}

WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
}

DEFAULT_EMOTION_MESSAGES = {
    "happy": {"text": "Thank you", "description": "A positive social cue"},
    "sad": {"text": "I feel sad", "description": "A sadness cue"},
    "angry": {"text": "I am upset", "description": "An anger cue"},
    "surprise": {"text": "That is surprising", "description": "A surprise cue"},
    "fear": {"text": "I feel nervous", "description": "A fear cue"},
    "disgust": {"text": "I do not like this", "description": "A dislike cue"},
    "neutral": {"text": "", "description": "No clear emotional cue"},
    "uncertain": {"text": "", "description": "Low-confidence prediction"},
    "no_face": {"text": "", "description": "No face detected"},
}

DEFAULT_GESTURE_MESSAGES = {
    "nodding": {"text": "Yes", "display_name": "Nodding", "emoji": "Yes"},
    "head_shaking": {"text": "No", "display_name": "Head Shaking", "emoji": "No"},
    "hand_raising": {"text": "Excuse me", "display_name": "Hand Raising", "emoji": "Hand"},
}
