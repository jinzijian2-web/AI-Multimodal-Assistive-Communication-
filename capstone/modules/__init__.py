"""Core modules for the multimodal assistive communication system."""

from .database import DatabaseManager
from .emotion_recognition import EmotionRecognizer
from .gesture_recognition import GestureRecognizer
from .speech_to_text import SpeechToText
from .text_to_speech import TextToSpeech

__all__ = [
    "DatabaseManager",
    "EmotionRecognizer",
    "GestureRecognizer",
    "SpeechToText",
    "TextToSpeech",
]
