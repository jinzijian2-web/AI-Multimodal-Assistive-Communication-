# AI Multimodal Assistive Communication System

## GitHub Repository

Code repository:

```text
https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME
```

Please replace the link above with the actual GitHub repository URL after uploading the project.

---

## Project Overview

This project is a multimodal assistive communication system designed for users with hearing or speech impairments. It integrates facial expression recognition, gesture recognition, speech-to-text conversion, text-to-speech output, and customizable gesture-to-text mapping into a web-based platform.

The system uses computer vision and speech processing technologies to support daily communication. It can recognize facial expressions, detect simple gestures, transcribe speech into text, convert text into speech, and allow users to customize gesture-based output messages.

---

## Main Features

- Facial expression recognition using a CNN model trained on the FER2013 dataset
- Gesture recognition using MediaPipe for nodding, head shaking, and hand raising
- Speech-to-text transcription using OpenAI Whisper
- Text-to-speech output using gTTS
- Custom gesture-to-text mapping stored in SQLite
- Web-based user interface built with Flask, HTML, CSS, and JavaScript
- Local processing for core recognition modules
- Low-cost deployment using a standard laptop, webcam, and microphone

---

## Recommended Environment

This project is recommended to run with:

```text
Python 3.11
Windows 10 or Windows 11
Webcam or integrated camera
Microphone
FFmpeg installed and added to system PATH
```

Python 3.11 is recommended because some dependencies, especially TensorFlow and MediaPipe, may not work correctly with newer Python versions such as Python 3.13.

---

## Required Python Packages

The required packages are listed in `requirements.txt`.

Recommended `requirements.txt`:

```txt
flask
flask-cors
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.1.78
mediapipe==0.10.7
openai-whisper
gTTS
pydub
Pillow
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## FFmpeg Requirement

The speech-to-text module uses OpenAI Whisper, which requires FFmpeg to process audio files.

### Install FFmpeg on Windows

One recommended method is to install FFmpeg using `winget`:

```powershell
winget install Gyan.FFmpeg
```

After installation, close and reopen the terminal or restart VS Code. Then check whether FFmpeg is available:

```powershell
ffmpeg -version
```

If the command shows FFmpeg version information, the installation is successful.

---

## Model File

The facial expression recognition module requires a trained model file:

```text
models/emotion_model.h5
```

Make sure this file exists before running the project.

If the model file is not included in the GitHub repository due to file size limitations, it should be provided separately and placed in the `models/` folder.

---

## How to Run the Project

### Step 1: Clone or download the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

If the project is downloaded as a ZIP file, extract it and open the extracted folder.

---

### Step 2: Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv venv
```

---

### Step 3: Activate the virtual environment

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(venv)
```

---

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

If the network connection is unstable, run:

```powershell
pip install -r requirements.txt --default-timeout=1000 --retries 10
```

---

### Step 5: Check FFmpeg

```powershell
ffmpeg -version
```

If FFmpeg is not recognized, install it and restart the terminal.

---

### Step 6: Run the Flask application

```powershell
python app.py
```

If the application starts successfully, the terminal will show:

```text
Running on http://127.0.0.1:5000
```

---

### Step 7: Open the web interface

Open a web browser and visit:

```text
http://127.0.0.1:5000
```

The live video stream can also be tested directly at:

```text
http://127.0.0.1:5000/video_feed
```

---

## Camera Troubleshooting

If the camera does not show video, check the following:

1. Make sure no other application is using the camera, such as Zoom, Teams, WeChat, Camera, or another browser page.
2. Check Windows camera permission:

```text
Settings → Privacy & security → Camera
```

Make sure the following options are enabled:

```text
Camera access: On
Let apps access your camera: On
Let desktop apps access your camera: On
```

3. For Lenovo laptops, check whether the camera privacy switch or Lenovo Privacy Mode is enabled. If it is enabled, disable it first.
4. If the camera has a physical privacy cover, make sure it is open.
5. Test the camera in Windows Camera before running the project.

The project uses OpenCV with DirectShow on Windows:

```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

This improves camera compatibility on Windows.

---

## Speech-to-Text Troubleshooting

If speech recognition fails with an error such as:

```text
[WinError 2] The system cannot find the file specified
```

it usually means FFmpeg is not installed or is not added to the system PATH.

Install FFmpeg and verify it with:

```powershell
ffmpeg -version
```

Then restart VS Code and run the project again.

---

## Basic Usage

1. Open the web page at `http://127.0.0.1:5000`.
2. The live camera stream will appear in the recognition panel.
3. Facial expressions and gestures will be recognized in real time.
4. Use the text input box to enter a message and click `Speak` to generate speech.
5. Use the `Speech to Text` function to record audio and transcribe it into text.
6. Use the custom gesture mapping section to define gesture-related output messages.
7. Quick phrases can be used for common communication expressions.

---

## Project Structure

A typical project structure is shown below:

```text
multimodal_system_fixed/
│
├── app.py
├── config.py
├── index.html
├── requirements.txt
├── README.md
│
├── models/
│   └── emotion_model.h5
│
├── modules/
│   ├── database.py
│   ├── emotion_recognition.py
│   ├── gesture_recognition.py
│   ├── speech_to_text.py
│   └── text_to_speech.py
│
├── database/
│   └── mappings.db
│
├── tts_cache/
│
└── static/
```

---

## Notes

- The system should be run with Python 3.11.
- The facial expression model file must be placed in the `models/` folder.
- FFmpeg is required for the speech-to-text module.
- gTTS may require an internet connection when generating new speech audio.
- The camera must be enabled at both the system level and hardware level.
- This project is a prototype for academic demonstration and is not a clinically certified assistive device.

---

## Author

Jin Zijian

## Supervisor

Dr. Nasser Mustafa
