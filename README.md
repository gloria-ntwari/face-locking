# Face Recognition & Face Locking -- 5-Point Pipeline (Windows)

This project implements a **real-time face recognition and face locking
system** using **5-point facial landmarks**, **face alignment**, and
**ArcFace embeddings**. It is designed to run cleanly on **Windows +
Python 3.12**.

The system can: - Detect multiple faces - Recognize identities - **Lock
onto a target face** - Track actions (head movement, blink, smile) - Log
actions to a history file

------------------------------------------------------------------------

## 📁 Project Structure (Core)

    Face_recognition/
    │
    ├── .venv/
    ├── buffalo_l/
    ├── data/
    │   ├── enroll/
    │   └── db/
    │
    ├── models/
    │   └── embedder_arcface.onnx
    │
    ├── src/
    │   ├── __init__.py
    │   ├── detect.py
    │   ├── recognize.py
    │   ├── haar_5pt.py
    │   ├── align.py
    │   ├── embed.py
    │   ├── enroll.py
    │   ├── evaluate.py
    │   ├── face_lock.py
    │   ├── action_detection.py
    │   ├── history_logger.py
    │   └── camera.py
    │
    ├── init_project.py
    └── README.md

------------------------------------------------------------------------

## 🐍 Python Version

    Python 3.12.4

------------------------------------------------------------------------

## 🔧 Setup (Windows)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python numpy onnxruntime mediapipe insightface
```

### MediaPipe Fix

``` powershell
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

------------------------------------------------------------------------

## 🧠 ArcFace Model Fix

``` powershell
Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
```

------------------------------------------------------------------------

## ▶️ Run Face Locking System

``` powershell
python -m src.detect
```

This launches: - Camera - Face recognition - Face locking - Action
logging

------------------------------------------------------------------------

## 🔒 Face Locking Summary

-   Locks onto a target identity
-   Tracks movement, blink, smile
-   Logs actions to:

```{=html}
<!-- -->
```
    history/<identity>_history_YYYYMMDDHHMMSS.txt

------------------------------------------------------------------------

## ❗ Common Errors

**ModuleNotFoundError** - Ensure `src/__init__.py` exists - Always run
with `python -m src.detect`

**MediaPipe error**

``` powershell
pip install mediapipe==0.10.9
```

------------------------------------------------------------------------

## 🎯 Notes

-   Uses 5-point landmarks
-   CPU-only ONNX
-   Windows-optimized

------------------------------------------------------------------------

## 🚀 Future Work

-   FAISS search
-   GUI
-   Multi-face locking
-   Database logging
