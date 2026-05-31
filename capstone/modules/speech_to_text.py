"""Speech-to-text module based on OpenAI Whisper."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import whisper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEECH_CONFIG


class SpeechToText:
    """Transcribe audio files or uploaded audio bytes using Whisper."""

    def __init__(self, model_size: Optional[str] = None):
        self.model_size = model_size or SPEECH_CONFIG["whisper_model"]
        print(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file."""
        language = language or SPEECH_CONFIG.get("speech_language", "en")
        result = self.model.transcribe(audio_path, language=language)
        return result.get("text", "").strip()

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm", language: Optional[str] = None) -> str:
        """Transcribe uploaded audio bytes."""
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            return self.transcribe(temp_path, language=language)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_from_microphone(self, duration: int = 3, language: Optional[str] = None) -> str:
        """Record audio from the microphone and transcribe it."""
        import pyaudio
        import wave

        audio_format = pyaudio.paInt16
        channels = 1
        rate = 16000
        chunk = 1024

        recorder = pyaudio.PyAudio()
        stream = recorder.open(
            format=audio_format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        print(f"Recording for {duration} seconds...")
        frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * duration))]
        stream.stop_stream()
        stream.close()
        recorder.terminate()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            with wave.open(temp_path, "wb") as wave_file:
                wave_file.setnchannels(channels)
                wave_file.setsampwidth(recorder.get_sample_size(audio_format))
                wave_file.setframerate(rate)
                wave_file.writeframes(b"".join(frames))
            return self.transcribe(temp_path, language=language)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    stt = SpeechToText()
    print("Testing microphone transcription...")
    print(stt.transcribe_from_microphone(duration=3))
