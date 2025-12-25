"""Simple YOLOv8 inference viewer (PyQt5).

- Choose a model file (.pt) and camera index.
- Live preview with bounding boxes and class labels.
- Reuses the camera dropdown pattern from main.py.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Avoid OpenMP runtime conflicts between OpenCV and PyTorch on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_TORCH_IMPORT_ERROR: Optional[Exception] = None
try:
    import torch  # noqa: F401
except Exception as exc:  # noqa: BLE001
    _TORCH_IMPORT_ERROR = exc

import psutil
from PyQt5 import QtCore, QtGui, QtWidgets

# cv2 is imported lazily after torch preload to avoid OpenMP conflicts on Windows.
cv2 = None

WINDOW_TITLE = "View Inference"
try:
    build_source = Path(__file__).resolve()
    BUILD_DATE = datetime.fromtimestamp(build_source.stat().st_mtime).strftime("%d/%m/%Y")
except OSError:
    BUILD_DATE = datetime.now().strftime("%d/%m/%Y")
ICON_FILE = Path("runIcon.ico")
CLASS_COLORS_PATH = Path("class_colors.json")
CAPTURE_DIR = Path("captures")
INFERENCE_DIR = CAPTURE_DIR / "inference"
DEFAULT_CLASS_COLORS = [
    (0, 255, 0),     # green
    (0, 200, 255),   # yellow-ish
    (0, 128, 255),   # orange
    (255, 0, 0),     # blue
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (180, 105, 255), # pink
    (128, 0, 255),   # purple
    (0, 255, 255),   # light blue
    (0, 255, 128),   # mint
]


def load_class_colors(path: Path = CLASS_COLORS_PATH):
    """Load class color palette from JSON; fall back to defaults on errors."""
    if not path.exists():
        return list(DEFAULT_CLASS_COLORS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        colors = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                r, g, b = (int(max(0, min(255, v))) for v in item)
                colors.append((r, g, b))
        return colors or list(DEFAULT_CLASS_COLORS)
    except (json.JSONDecodeError, OSError, ValueError):
        return list(DEFAULT_CLASS_COLORS)


def load_app_icon() -> Optional[QtGui.QIcon]:
    """Load the app icon (runIcon.ico) with PyInstaller support."""
    try:
        base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path.cwd()
    except Exception:
        base = Path.cwd()
    candidate_paths = [
        base / ICON_FILE.name,
        Path(sys.executable).parent / ICON_FILE.name if getattr(sys, "frozen", False) else None,
    ]
    for path in candidate_paths:
        if path and path.exists():
            icon = QtGui.QIcon(str(path))
            if not icon.isNull():
                return icon
    return None


def create_splash() -> QtWidgets.QSplashScreen:
    """Create a simple splash screen so the user sees activity on startup."""
    splash_pix = QtGui.QPixmap(460, 420)
    splash_pix.fill(QtGui.QColor("#f0f0f0"))
    painter = QtGui.QPainter(splash_pix)
    painter.setPen(QtGui.QColor("#000000"))
    painter.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
    painter.drawText(splash_pix.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, WINDOW_TITLE)

    text_y = 120

    painter.setFont(QtGui.QFont("Segoe UI", 12))
    text_height = max(40, splash_pix.height() - text_y - 20)
    footer_text = f"Loading {WINDOW_TITLE}...\nBuild: {BUILD_DATE} | JShade.co.uk"
    painter.drawText(
        0,
        text_y,
        splash_pix.width(),
        text_height,
        QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
        footer_text,
    )
    painter.end()

    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    QtWidgets.QApplication.processEvents()
    return splash


def class_color(class_id: int, palette):
    """Pick a color for a class id from the provided palette."""
    if not palette:
        return 0, 255, 0
    return palette[class_id % len(palette)]


def draw_text_with_bg(
    img,
    text: str,
    org: tuple,
    font_scale: float = 0.5,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    bg_color: tuple = (0, 0, 0),
    font=None,
) -> None:
    """Render text with a solid background for readability."""
    global cv2
    if cv2 is None:
        # Import here to avoid DLL conflicts between OpenCV and PyTorch on Windows.
        import cv2 as _cv2
        cv2 = _cv2
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 2, y - text_h - baseline - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def enumerate_cameras(max_index: int = 5):
    """Probe a handful of indices to find available cameras."""
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found


def open_camera(index: int):
    """Open a camera index and return a capture handle."""
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    cap = cv2.VideoCapture(index)
    return cap


class InferenceWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        # Tracking fields for camera, model, and UI state.
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = 0
        self.model = None
        self.model_names: Dict[int, str] = {}
        self.device_choice = "cpu"
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.current_frame = None
        self.last_display_frame = None
        self.telemetry_visible = True
        self._gpu_available = False
        self._torch_ok = False
        self.class_colors = load_class_colors()
        # Detect system capabilities and pre-load heavy libs.
        self._init_gpu_info()
        self._preload_torch()
        self._import_cv2()
        if not self._torch_ok:
            # Exit cleanly after showing the error message.
            QtCore.QTimer.singleShot(0, QtWidgets.QApplication.instance().quit)
            return

        # Top nav with menu + camera selector + model picker.
        self.menu_bar = QtWidgets.QMenuBar()
        file_menu = self.menu_bar.addMenu("File")
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.qApp.quit)
        help_menu = self.menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        keys_action = help_menu.addAction("Key Bindings")
        keys_action.triggered.connect(self.show_key_bindings)
        self.menu_bar.setStyleSheet(
            "QMenuBar { background: #e6e6e6; color: black; }"
            "QMenuBar::item { background: #e6e6e6; padding: 4px 10px; }"
            "QMenuBar::item:selected { background: #d6d6d6; }"
            "QMenu { background: #f0f0f0; color: black; }"
        )

        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.camera_selector.setMinimumWidth(120)
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        self.populate_camera_selector()
        self.camera_selector.setStyleSheet(
            "QComboBox { background: #f0f0f0; color: black; }"
            "QComboBox QAbstractItemView { background: #f0f0f0; color: black; }"
        )

        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText("Select model (.pt)")
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["CPU", "GPU (auto)"])
        self.device_combo.currentIndexChanged.connect(self.change_device)

        nav_bar = QtWidgets.QWidget()
        nav_bar.setStyleSheet("background: #e6e6e6;")
        nav_layout = QtWidgets.QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        nav_layout.addWidget(self.menu_bar)
        nav_layout.addStretch()
        nav_layout.addWidget(QtWidgets.QLabel("Camera:"))
        nav_layout.addWidget(self.camera_selector)
        nav_layout.addStretch()
        nav_layout.addWidget(QtWidgets.QLabel("Device:"))
        nav_layout.addWidget(self.device_combo)
        nav_layout.addStretch()
        nav_layout.addWidget(QtWidgets.QLabel("Model:"))
        nav_layout.addWidget(self.model_path_edit)
        nav_layout.addWidget(browse_btn)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_label.setText("Select a model to start.")
        self.capture_button = QtWidgets.QPushButton("Capture (C)")
        self.capture_button.clicked.connect(self.capture_screenshot)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(nav_bar)
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # Timer drives the capture loop without blocking the GUI.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

        self.capture_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("C"), self)
        self.capture_shortcut.activated.connect(self.capture_screenshot)
        self.telemetry_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("H"), self)
        self.telemetry_shortcut.activated.connect(self.toggle_telemetry)
        self.quit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self)
        self.quit_shortcut.activated.connect(QtWidgets.qApp.quit)

        self.cap = open_camera(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.status_label.setText("Unable to open camera.")

    def populate_camera_selector(self):
        """Refresh the camera dropdown with available devices."""
        cams = enumerate_cameras()
        if not cams:
            cams = [self.camera_index]
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for idx in cams:
            self.camera_selector.addItem(f"Camera {idx}", idx)
        if self.camera_selector.findData(self.camera_index) == -1:
            self.camera_selector.addItem(f"Camera {self.camera_index}", self.camera_index)
        current_idx = self.camera_selector.findData(self.camera_index)
        if current_idx >= 0:
            self.camera_selector.setCurrentIndex(current_idx)
        self.camera_selector.blockSignals(False)

    def change_camera(self, idx):
        """Swap to a new camera index when the user changes the dropdown."""
        new_index = self.camera_selector.itemData(idx)
        if new_index is None or new_index == self.camera_index:
            return
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = open_camera(new_index)
        self.camera_index = new_index
        if not self.cap or not self.cap.isOpened():
            self.status_label.setText(f"Unable to open camera {new_index}.")
        else:
            self.status_label.setText(f"Switched to camera {new_index}.")

    def browse_model(self):
        """Open a file dialog and load the selected model."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model file", filter="Model Files (*.pt)")
        if path:
            self.model_path_edit.setText(path)
            self.load_model(Path(path))

    def load_model(self, path: Path):
        """Load a YOLO model with defensive error handling."""
        if not path.exists():
            self._show_error("Missing model", f"Model file not found:\n{path}")
            return
        if not self._torch_ok:
            self._show_error(
                "Dependency missing",
                "Install ultralytics and torch in this environment.",
            )
            return
        try:
            self._add_torch_dll_dir()
            from ultralytics import YOLO
        except Exception as exc:
            self._show_error(
                "Dependency missing",
                "Install ultralytics and torch in this environment.",
                exc,
            )
            return
        try:
            self.model = YOLO(str(path))
            self.model_names = self.model.names
            self.status_label.setText(f"Loaded model: {path.name} | Device: {self.device_choice.upper()}")
        except Exception as exc:
            self._show_error("Model load failed", "Could not load model.", exc)
            self.model = None
            self.model_names = {}

    def update_frame(self):
        """Grab a frame, run inference if a model is loaded, and display results."""
        global cv2
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText("Failed to read frame.")
            return

        if self.model:
            try:
                # YOLO returns a list of results; we only need the first entry for a single frame.
                results = self.model(frame, verbose=False, imgsz=640, device=self.device_choice)[0]
                frame = self.draw_detections(frame, results)
            except Exception as exc:
                self.status_label.setText(f"Inference error: {exc}")

        # Overlay telemetry
        if self.telemetry_visible:
            self.update_fps()
            telemetry_lines = self.build_telemetry_lines()
            frame = self.draw_overlay_text(frame, telemetry_lines)

        # Convert to Qt so the QLabel can display the image.
        self.current_frame = frame
        self.last_display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def draw_detections(self, frame, results):
        """Draw boxes and labels for each detection."""
        global cv2
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return frame
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            color = class_color(int(k), self.class_colors)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            name = self.model_names.get(int(k), str(int(k)))
            label = f"{name} {c:.2f}"
            cv2.putText(frame, label, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

    def change_device(self, idx: int) -> None:
        choice = self.device_combo.currentText().lower()
        self.device_choice = "cpu" if "cpu" in choice else "0"
        if self.model:
            self.status_label.setText(f"Loaded model: {Path(self.model_path_edit.text()).name} | Device: {self.device_choice.upper()}")

    def show_about(self) -> None:
        QtWidgets.QMessageBox.information(self, "About", f"{WINDOW_TITLE}\nBuild date: {BUILD_DATE}")

    def show_key_bindings(self) -> None:
        html = """
<b>Key Bindings</b>
<pre>
C             Capture screenshot
H             Toggle telemetry
Q             Quit
</pre>
"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Key Bindings")
        msg.setText(html)
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def toggle_telemetry(self) -> None:
        """Show or hide the FPS/CPU/RAM overlay."""
        self.telemetry_visible = not self.telemetry_visible
        state = "shown" if self.telemetry_visible else "hidden"
        self.status_label.setText(f"Telemetry {state}.")

    def capture_screenshot(self) -> None:
        global cv2
        if self.last_display_frame is None:
            self.status_label.setText("No frame available yet.")
            return
        # Store screenshots under captures/inference for easy cleanup.
        INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = INFERENCE_DIR / f"capture_{timestamp}.jpg"
        success = cv2.imwrite(str(image_path), self.last_display_frame)
        if success:
            self.status_label.setText(f"Captured {image_path.name}.")
        else:
            self.status_label.setText("Failed to save capture.")

    def update_fps(self) -> None:
        """Maintain a smoothed FPS value so the overlay is stable."""
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            inst = 1.0 / dt
            self.fps = 0.8 * self.fps + 0.2 * inst if self.fps > 0 else inst
        self.last_frame_time = now

    def build_telemetry_lines(self):
        """Read current CPU/RAM/GPU usage for on-screen display."""
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu = "N/A"
        vram = "N/A"
        if self._gpu_available:
            try:
                # pynvml is optional; if it isn't available we fall back to N/A.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=r"The pynvml package is deprecated.*",
                    )
                    import pynvml  # type: ignore
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = f"{gpu_util}%"
                vram = f"{mem_info.used / (1024**2):.0f} / {mem_info.total / (1024**2):.0f} MB"
            except Exception:
                gpu = "N/A"
                vram = "N/A"
        return [
            f"FPS: {self.fps:.1f}",
            f"CPU: {cpu:.0f}%",
            f"RAM: {mem:.0f}%",
            f"GPU: {gpu}",
            f"VRAM: {vram}",
        ]

    def draw_overlay_text(self, frame, lines):
        """Draw multiple text lines with spacing so they don't overlap."""
        y = 20
        for line in lines:
            draw_text_with_bg(frame, line, (10, y), font_scale=0.6, color=(0, 255, 0), thickness=1, bg_color=(0, 0, 0))
            y += 24
        return frame

    def _init_gpu_info(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r"The pynvml package is deprecated.*",
                )
                import pynvml  # type: ignore
                pynvml.nvmlInit()
            self._gpu_available = True
        except Exception:
            self._gpu_available = False

    def _import_cv2(self) -> None:
        """Import cv2 after torch preload to reduce DLL conflicts."""
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2

    def _preload_torch(self) -> None:
        """Preload torch early to surface DLL issues immediately."""
        if _TORCH_IMPORT_ERROR is not None:
            self._torch_ok = False
            self._show_error(
                "Dependency missing",
                "Install ultralytics and torch in this environment.",
                _TORCH_IMPORT_ERROR,
            )
            return
        try:
            self._add_torch_dll_dir()
            import torch  # noqa: F401
            self._torch_ok = True
        except Exception as exc:
            self._torch_ok = False
            self._show_error(
                "Dependency missing",
                "Install ultralytics and torch in this environment.",
                exc,
            )

    def _show_error(self, title: str, message: str, exc: Optional[Exception] = None) -> None:
        details = f"{message}"
        if exc is not None:
            details = f"{message}\n\nDetails:\n{exc}"
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(details)
        msg.setInformativeText("")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        msg.exec_()

    def _add_torch_dll_dir(self) -> None:
        """Ensure torch DLL directory is on the search path (Windows)."""
        if os.name != "nt":
            return
        try:
            venv_root = Path(sys.executable).resolve().parent.parent
            torch_lib = venv_root / "Lib" / "site-packages" / "torch" / "lib"
            if torch_lib.exists():
                os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH','')}"
                os.add_dll_directory(str(torch_lib))
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    app_icon = load_app_icon()
    if app_icon:
        app.setWindowIcon(app_icon)
    splash = create_splash()

    window = InferenceWindow()
    if app_icon:
        window.setWindowIcon(app_icon)
    window.resize(960, 720)
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
