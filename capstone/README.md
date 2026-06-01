# Multimodal Assistive Communication System

This project is a web-based prototype for multimodal assistive communication. It integrates facial expression recognition, gesture recognition, speech-to-text conversion, text-to-speech synthesis, and customizable gesture-to-text mappings.

## Main Features

- Facial expression recognition using a CNN model trained on FER2013
- Gesture recognition using MediaPipe landmarks for nodding, head shaking, and hand raising
- Speech-to-text transcription using OpenAI Whisper
- Text-to-speech synthesis using gTTS
- Local SQLite database for custom gesture-to-text mappings
- English web interface

## Installation
This project is recommended to run with Python 3.11.

Before running the speech-to-text module, FFmpeg must be installed and added to the system PATH.

If the camera does not show video, please check whether the camera privacy switch, Lenovo Privacy Mode, or Windows camera permission is blocking the camera.
```bash
pip install -r requirements.txt
```

Whisper requires FFmpeg for browser-recorded audio files. Install FFmpeg before using the speech-to-text function.

## Run

```bash
python app.py
```

Then open the following address in a browser:

```text
http://127.0.0.1:5000
```

## Notes

The default model path is `models/emotion_model.h5`. If the model file is missing, the emotion module will run in a simple fallback mode.
