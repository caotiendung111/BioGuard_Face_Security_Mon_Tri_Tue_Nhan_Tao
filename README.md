# 🔐 DeepLock Pro - Advanced Biometric Face Authentication

<div align="center">

[![GitHub Actions CI](https://img.shields.io/github/actions/workflow/status/caotiendung111/bioguard-face-security/ci.yml?branch=main&logo=github&style=flat-square)](https://github.com/caotiendung111/bioguard-face-security/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&style=flat-square)](https://streamlit.io/)
[![Cryptography](https://img.shields.io/badge/Biometrics-AES--128--Fernet-success?logo=keybase&style=flat-square)](https://cryptography.io/)
[![Liveness](https://img.shields.io/badge/Liveness-Anti--Spoofing-orange?style=flat-square)](#)
[![Low Light](https://img.shields.io/badge/Night%20Shift-CLAHE%20AI-purple?style=flat-square)](#)

**A premium real-time face unlock system featuring Liveness Detection (Anti-spoofing), AES-128 Fernet Biometric Encryption, and a Vietnamese Voice Assistant.**

---

### 💠 Interactive Biometric Shield Dashboard
![DeepLock Pro Dashboard](assets/dashboard.png)

</div>

---

**DeepLock Pro** (originally named *FaceUnlock*) is a comprehensive, enterprise-grade real-time biometric face authentication solution. It integrates modern AI computer vision algorithms with secure, localized cryptographic structures to deliver a safe, fast, and intelligent authentication experience via local webcams or remote IP Cameras.

---

## 🏗️ System Architecture

The following diagram illustrates the flow of data through the biometric authentication pipeline, from raw frame capture to low-light enhancement, land-marking, liveness checks, cryptographic vector matching, and interactive voice welcome:

```mermaid
graph TD
    Frame[Webcam / IP Camera Video Frame] --> LightChk{Low Light?}
    LightChk -- Yes --> CLAHE[CLAHE Image Contrast Enhancer]
    LightChk -- No --> Detect[Face Detection & Landmarking - MediaPipe]
    CLAHE --> Detect
    Detect --> Liveness{Liveness Checks passed?}
    Liveness -- EAR/Pose/Anti-Spoofing -- -- No --> Block[Spoofing Alert / Audio Warning]
    Liveness -- Yes --> Embed[Extract 128-D Biometric Embedding]
    Embed --> Crypt[Symmetric Cryptographic Decryption - Fernet]
    DB[(Encrypted Database - faces/*.npy.enc)] --> Crypt
    Crypt --> Auth{Match Found?}
    Auth -- Yes --> Access[Grant Access & Audio Welcome - gTTS]
    Auth -- No --> Deny[Deny Access / Alert Logged]
```

---

## ⚡ Core Features

*   **🛡️ State-of-the-Art Anti-Spoofing (Liveness Detection)**:
    *   **Eye Aspect Ratio (EAR)**: Continuously analyzes eye-blinking patterns over 15 consecutive frames.
    *   **Head Pose Estimation**: Tracks 3D head orientation (Yaw, Pitch, Roll) to verify active user presence, successfully defending against static photos and video replay attacks.
*   **🔐 Biometric Encryption (AES Fernet 128-bit)**:
    *   Extracted 128-D face embeddings (via `face_recognition`) are symmetrically encrypted before writing to disk.
    *   Cryptographic keys are managed securely through `.env` files.
    *   **Automatic Migration**: Automatically scans, encrypts, and overrides any existing unsecured biometric files at runtime.
*   **🌙 Night Shift AI (CLAHE Low-Light Enhancement)**:
    *   Automatically measures average ambient lighting in real-time.
    *   Applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) on the Lightness channel of the LAB color space, improving face recognition sensitivity and accuracy by up to **2x** in dim environments.
*   **🗣️ Interactive Audio Feedback (Vietnamese Text-to-Speech)**:
    *   Generates runtime audio feedback using `gTTS` directly in memory streams (no disk I/O latency).
    *   Welcomes authenticated users by name and plays warnings for spoofing or shoulder-surfing behaviors.
*   **👥 Shoulder Surfing Detection (Ghost Mode)**:
    *   Scans the background for multiple faces and alerts the user if an unauthorized person is looking over their shoulder.

---

## 📂 Project Architecture

```text
FaceUnlock/
├── .github/workflows/       # GitHub Actions CI configurations
│   └── ci.yml
├── assets/                  # Media resources, UI graphics, and screenshots
├── faces/                   # Encrypted biometric database (.npy.enc)
├── tests/                   # Automated unit and integration tests
│   └── test_encryption.py
├── app.py                   # Central Streamlit web dashboard
├── main.py                  # Core terminal-based camera processing loop
├── face_utils.py            # AI inference, Liveness detection, CLAHE, and Audio utilities
├── requirements.txt         # Production library dependencies
├── requirements-dev.txt     # Development and testing dependencies
└── .env.example             # Template for environment configuration
```

---

## 🛠️ Installation Guide

### 1. Setup Virtual Environment
Run the following commands to initialize and activate a Python virtual environment:

```powershell
# Navigate into the project folder
cd FaceUnlock

# Create a Python 3.12 virtual environment
py -3.12 -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Upgrade package manager and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the `.env.example` file to `.env` to initialize key generation:
```powershell
copy .env.example .env
```

---

## 🚀 Running the Project

### Streamlit Web Dashboard
Launch the web interface providing real-time telemetry and a biometric control dashboard:
```powershell
streamlit run app.py
```

### Terminal Interface
Run the main loop in the console with support for USB webcams or IP cameras:

*   **Launch via default USB webcam / Virtual Camera:**
    ```powershell
    python main.py --camera 1
    ```
*   **Launch via IP Camera RTSP/HTTP Stream:**
    Ensure your device and PC are on the same local network, fetch the streaming URL, and run:
    ```powershell
    python main.py --url http://192.168.1.5:8080/video
    ```

**Console Controls:**
*   Press `r` to register a new user face (input their name in the console).
*   Press `q` to quit the application stream.

---

## 🧪 Automated Testing
Run automated unit tests to verify system security, cryptographic standards, and computer vision filters:
```powershell
pytest
```
The test suite validates:
*   Fernet key generation and secure loading.
*   128-D vector encryption parity (tolerance error < 1e-6).
*   Automatic database migration for legacy unencrypted files.
*   CLAHE color space conversion stability.

---

## 📈 Known Limitations & Future Improvements

To show engineering foresight, we document the core physical limitations of the local model deployment and the roadmap for enterprise scaling:

- **CPU-bound Inference Bottleneck**: Model inference for face land-marking (MediaPipe) and vector mapping (`dlib`/`face_recognition`) is CPU-heavy on local machines. In a production environment, this should be offloaded to a GPU instance or run asynchronously via a task broker queue (e.g., Celery with Redis).
- **Physical Device Dependency for Audio**: Local playback of Text-to-Speech audio uses `pygame`, which queries local sound devices (like ALSA on Linux or CoreAudio on macOS). This makes headless Docker deployments throw output device warnings. An enterprise improvement would be serving the TTS audio payload stream as an base64 encoded audio asset inside a REST API response for browser-side rendering.
- **Flat-file Biometric Database**: The localized storage of encrypted vectors in flat `.npy.enc` files works well for small groups of users but is inefficient for large-scale operations. A future improvement is migrating the vectors to a dedicated vector database index (such as **ChromaDB**, **Milvus**, or **pgvector** in PostgreSQL) to support fast Cosine Similarity searches on thousands of entries.
- **Single-user Bias**: The liveness checks are optimized for validating one subject standing in front of the lens. Support for group/multi-person authentication requires multi-threading of liveness state machines across active face bounding box IDs.
