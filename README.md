# OpenCV capture, labeling, training, and inference for YOLO object detection
## Overview

This project implements a **lightweight image capture, annotation, training, and inference toolkit** for object-detection workflows. The system is designed to operate in a **controlled, offline imaging setup** and to integrate directly with OpenCV-based workflows and modern deep-learning pipelines (e.g. YOLOv8).

The tool combines:
- Direct camera capture
- Interactive bounding-box annotation
- Structured dataset export
- Basic dataset validation utilities
- A training companion for YOLOv8
- A live inference viewer for quick model checks

The emphasis is on **simplicity, transparency, and reproducibility**, rather than real-time performance or industrial deployment.

---

## System Architecture

The system is structured as a small number of loosely coupled modules:

1. Capture Module  
2. Annotation UI  
3. Dataset Storage Layer  
4. Dataset Audit Utilities  
5. Training Companion App  
6. Inference Viewer App  

Each module is designed to be camera-agnostic and file-system-based to minimise external dependencies.

---

## Dependencies

- Python 3.12 (recommended to use prebuilt wheels for NumPy and OpenCV)
- pip
- OpenCV Python bindings (`opencv-python`, which installs `numpy`)
- PyQt5 (GUI toolkit for capture, training, and inference apps)
- Ultralytics (`ultralytics`) for YOLOv8 training and inference
- PyTorch (`torch`) for model loading and inference (required by Ultralytics)
- psutil (system telemetry in the inference viewer)
- Optional: `pynvml` for NVIDIA GPU telemetry (inference viewer)
- Optional: `pyinstaller` for packaging into a Windows executable

### Recommended local setup
- Create and use a dedicated venv named `.venv312` to avoid interpreter mismatches: `py -3.12 -m venv .venv312` then `.\.venv312\Scripts\activate`.
- Install packages inside it: `pip install --upgrade pip` then `pip install --only-binary=:all: "numpy<2.3.0" "opencv-python<4.13" pyqt5 ultralytics`.
- VS Code users: set the interpreter to `.venv312/Scripts/python.exe` (workspace setting already included in `.vscode/settings.json`). Terminals still need `.\.venv312\Scripts\activate` before running `python main.py`.

### Install Python 3.12 on Windows
1. Download the official Windows installer for Python 3.12.
2. During setup, tick **Add Python to PATH** and keep the Windows launcher (enables `py`).

### Create a virtual environment and install OpenCV (PowerShell)
```powershell
# From the repository root
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate

# Upgrade pip and install wheel-only builds to avoid compiling
pip install --upgrade pip
pip install --only-binary=:all: "numpy<2.3.0" "opencv-python<4.13" pyqt5 ultralytics torch psutil pynvml
```

---

## Image Capture Module

### Functionality
- Acquires frames from a connected camera using OpenCV (`cv2.VideoCapture`)
- Displays a live preview window for framing and focus checks
- Captures single frames on user input
- Writes image files directly to disk

### Design Considerations
- No assumptions are made about camera type (laptop webcam or external USB camera)
- Capture settings (resolution, camera index) are configurable
- Capture is explicitly user-triggered to avoid near-duplicate frames

### Output
- Images are saved using a deterministic naming scheme encoding:
  - capture date/time
  - tool or part identifier
  - sequential index
- Basic capture metadata is recorded separately for traceability

---

## Annotation Interface

### Annotation Model
- Object detection using axis-aligned bounding boxes
- Each bounding box is associated with a single defect class
- Multiple defects per image are supported
- Images with no defects are explicitly allowed

### User Interface
- Implemented using OpenCV window callbacks
- Mouse input:
  - click-and-drag to draw bounding boxes
- Keyboard input:
  - numeric keys for class assignment
  - undo last annotation
  - save and advance to next image

The UI is intentionally minimal to reduce annotation time and cognitive load.

---

## Label Format

Annotations are exported in **YOLO object detection format**, with one text file per image:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- All coordinates are normalised to image dimensions
- `class_id` corresponds to a fixed class list defined in `classes.txt`

This format was selected for compatibility with common training frameworks and ease of validation.

---

## Dataset Storage Layout

The dataset is stored entirely on disk using a transparent directory structure:

```
captures/
  capture_YYYYMMDD_HHMMSS_xxx.jpg
  null/                 # null/removed captures
classes.txt             # class names (one per line)
class_colors.json       # optional RGB palette
```


### Design Rationale
- Images are not stored in a database to avoid unnecessary I/O overhead
- Labels and metadata are human-readable
- The dataset can be inspected, versioned, or transferred without specialised tooling

---

## Dataset Integrity and Validation

A dataset audit step is included to ensure consistency before training.

Validation checks include:
- One-to-one correspondence between images and label files
- Bounding box values within valid ranges
- Detection of empty or malformed label files
- Summary statistics:
  - images per class
  - bounding boxes per class
  - distribution across tools or parts

This step is intended to catch annotation errors early and prevent silent failures during training.

---

## Integration with Training Pipelines

The generated dataset is intended to be consumed directly by object-detection training frameworks.

Key integration features:
- YOLO-compatible label format
- Deterministic filenames for reproducible splits
- Support for dataset partitioning based on tool, part geometry, or capture session

The capture and annotation codebase is deliberately decoupled from any specific training implementation, while the companion tools provide a short path into YOLOv8.

---

## Training Companion (YOLOv8)

The training UI (`train_model.py`) provides a guided path to start YOLOv8 training without writing scripts:

- Select a dataset folder and `classes.txt`
- Auto-build a temporary YOLO dataset structure and YAML
- 80/20 train/validation split with deterministic shuffling
- Launch the Ultralytics CLI with configurable model size, image size, epochs, and device
- Live logs with basic ETA parsing

The prepared dataset lives under `.yolo_training_cache/` inside the selected dataset folder.

---

## Inference Viewer

The inference UI (`run_inference.py`) provides a live camera preview with YOLOv8 detections:

- Select a camera and model file
- Run inference on live frames
- Render bounding boxes, class labels, and confidences
- Switch CPU/GPU (if available) from the UI

This tool is intended for quick sanity checks and demoing models, not for production deployment.

---

## Limitations

- Annotation is fully manual
- Inference is for visualization and validation only (not optimized for production)
- No automated defect proposal or pre-labelling
- Image quality and defect visibility depend on external hardware and lighting

These constraints are intentional to keep the system lightweight and maintainable.

---

## Intended Use

This tool is intended for:
- Rapid dataset generation for defect detection experiments
- Prototyping object-detection pipelines
- Controlled data collection in laboratory or bench-top environments

It is not intended for production inspection systems or safety-critical use.

---

## Useful Commands

### Run (dev)
```powershell
.\.venv312\Scripts\activate
python main.py --camera 0
```

### Run with explicit resolution
```powershell
.\.venv312\Scripts\activate
python main.py --camera 0 --width 1280 --height 720
```

### Train (YOLOv8 UI)
```powershell
.\.venv312\Scripts\activate
python train_model.py
```

### Inference (YOLOv8 UI)
```powershell
.\.venv312\Scripts\activate
python run_inference.py
```

### Build (PyInstaller, windowless)
```powershell
.\.venv312\Scripts\activate
pyinstaller --name OpenCVCapture --noconfirm --onedir -w main.py `
  --icon "programLogo.ico" `
  --hidden-import sip `
  --add-data "classes.txt;." `
  --add-data "class_colors.json;." `
  --add-data "captures;captures" `
  --add-data "programLogo.ico;." `
  --add-data "$Env:VIRTUAL_ENV\Lib\site-packages\PyQt5\Qt5\plugins;PyQt5\Qt5\plugins"
```
Output lives in `dist/OpenCVCapture/`. Run from there: `.\OpenCVCapture.exe --camera 0`.

### Clean build artifacts
```powershell
Remove-Item -Recurse -Force build, dist, OpenCVCapture.spec
```

### Key bindings

**Main window**
```
C             Capture frame
Q             Quit
LEFT/RIGHT    Prev/next (inspect mode)
DELETE        Delete current file (inspect mode)
E             Edit current file (inspect mode)
ESC           Exit inspect mode
```

**Labeler window**
```
Drag (LMB)    Draw box
0-9           Choose class
Enter         Apply class to selected box
Left/Right    Select box
Up/Down       Cycle class for selected box
U / Z         Undo last box
S             Save labels
N             Mark null
Q / ESC       Cancel labeling
Scroll        Zoom
Middle drag   Pan
```

**Inference window**
```
C             Capture screenshot
H             Toggle telemetry overlay
Q             Quit
```

**Training window**
```
Ctrl+Q        Quit
```
