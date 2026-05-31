"""Text-to-speech module based on gTTS with local audio caching."""

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

from gtts import gTTS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEECH_CONFIG, TTS_CACHE_DIR


class TextToSpeech:
    """Generate speech audio files from text and cache common outputs."""

    def __init__(self, language: Optional[str] = None, slow: Optional[bool] = None, cache_dir: Optional[str] = None):
        self.language = language or SPEECH_CONFIG["tts_language"]
        self.slow = SPEECH_CONFIG["tts_slow"] if slow is None else slow
        self.cache_dir = Path(cache_dir or TTS_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Text-to-speech module initialized with language={self.language}.")

    def synthesize(self, text: str, language: Optional[str] = None) -> str:
        """Create or reuse a cached MP3 file and return its absolute path."""
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("No text was provided for speech synthesis.")

        language = language or self.language
        cache_key = hashlib.sha256(f"{language}|{self.slow}|{clean_text}".encode("utf-8")).hexdigest()[:24]
        audio_path = self.cache_dir / f"tts_{cache_key}.mp3"
        if not audio_path.exists():
            tts = gTTS(text=clean_text, lang=language, slow=self.slow)
            tts.save(str(audio_path))
        return str(audio_path)

    def speak(self, text: str, language: Optional[str] = None) -> str:
        """Compatibility wrapper that synthesizes speech and returns the audio path."""
        return self.synthesize(text, language=language)

    def speak_emotion(self, emotion: str, message: str) -> str:
        """Generate speech for an emotion recognition result."""
        return self.synthesize(f"Detected emotion: {emotion}. {message}")

    def speak_gesture(self, gesture_name: str, message: str) -> str:
        """Generate speech for a gesture recognition result."""
        return self.synthesize(f"Detected gesture: {gesture_name}. {message}")


if __name__ == "__main__":
    tts = TextToSpeech()
    path = tts.synthesize("Welcome to the multimodal assistive communication system.")
    print(f"Generated audio file: {path}")
