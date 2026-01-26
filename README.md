# Face Recognition – 5-Point Pipeline (Windows)

This project implements a **face recognition pipeline using 5-point facial landmarks**, face alignment, and ArcFace embeddings. It is structured to work cleanly on **Windows + Python 3.12**.

This README reflects fixes for the following issues you encountered:

* `mediapipe` missing `solutions`
* `ModuleNotFoundError: src.haar_5pt`
* Incorrect ONNX model paths

---

## 📁 Project Structure

```
Face_recognition/
│
├── .venv/                  # Python virtual environment
├── book/                   # Notes / documentation
├── buffalo_l/              # InsightFace detection models
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   ├── det_10g.onnx
│   ├── genderage.onnx
│   └── w600k_r50.onnx
│
├── data/
│   ├── enroll/             # Face images for enrollment
│   └── db/                 # Saved embeddings
│
├── models/
│   └── embedder_arcface.onnx
│
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── detect.py
│   ├── landmarks.py
│   ├── align.py
│   ├── embed.py
│   ├── enroll.py
│   ├── recognize.py
│   ├── evaluate.py
│   └── haar_5pt.py
│
├── init_project.py
└── README.md
```

---

## 🐍 Python Version

```powershell
python --version
```

Expected:

```
Python 3.12.4
```

---

## 🔧 Virtual Environment Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 📦 Install Dependencies

```powershell
pip install --upgrade pip
pip install opencv-python numpy onnxruntime mediapipe insightface
```

### ⚠️ MediaPipe Fix (IMPORTANT)

If you see:

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

Run:

```powershell
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

Verify:

```powershell
python - << EOF
import mediapipe as mp
print(mp.solutions)
EOF
```

---

## 🧠 Fix: ArcFace Model Location

Your ArcFace model exists here:

```
buffalo_l/w600k_r50.onnx
```

Copy it correctly:

```powershell
Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
```

❌ Wrong (causes error):

```powershell
Copy-Item w600k_r50.onnx models\embedder_arcface.onnx
```

---

## 🛠 Fix: Python Package Imports

Ensure `src/__init__.py` exists:

```powershell
New-Item src\__init__.py -ItemType File
```

All internal imports now use **relative imports**:

```python
from .haar_5pt import Haar5ptDetector, align_face_5pt
```

And scripts must be run as **modules**:

✅ Correct:

```powershell
python -m src.align
```

❌ Wrong:

```powershell
python src/align.py
```

---

## ▶️ Running Each Stage

### 1️⃣ Facial Landmarks (5-point)

```powershell
python -m src.landmarks
```

### 2️⃣ Face Alignment

```powershell
python -m src.align
```

### 3️⃣ Generate Face Embeddings

```powershell
python -m src.embed
```

### 4️⃣ Enroll Faces

```powershell
python -m src.enroll
```

### 5️⃣ Recognize Faces

```powershell
python -m src.recognize
```

### 6️⃣ Evaluation

```powershell
python -m src.evaluate
```

---

## 📷 Camera Test

```powershell
python -m src.camera
```

Press `Q` to quit.

---

## 🧪 Common Errors & Fixes

### ❌ `ModuleNotFoundError: src.haar_5pt`

✔ Fix:

* Ensure `src/haar_5pt.py` exists
* Ensure `src/__init__.py` exists
* Run using `python -m src.<module>`

---

### ❌ `mediapipe has no attribute solutions`

✔ Fix:

```powershell
pip install mediapipe==0.10.9
```

---

### ❌ ONNX model not found

✔ Fix path:

```powershell
models/embedder_arcface.onnx
```

---

## 🎯 Notes

* This pipeline uses **5-point Haar landmarks**, not full 468-point mesh
* ONNX models are CPU-safe
* Works on **Windows + Python 3.12**

---

## 🚀 Next Steps

* Add face tracking
* Optimize embedding comparison (FAISS)
* Add GUI (Tkinter / PyQt)
* Export embeddings to database

---

✅ If you want, I can also:

* Review your `haar_5pt.py`
* Add logging
* Convert this to a real-time system
* Package it as an installable module

Just tell me 👍
