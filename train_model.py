"""YOLOv8 training companion app (PyQt5)."""

import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets


WINDOW_TITLE = "Train Model"
try:
    build_source = Path(__file__).resolve()
    BUILD_DATE = datetime.fromtimestamp(build_source.stat().st_mtime).strftime("%d/%m/%Y")
except OSError:
    BUILD_DATE = datetime.now().strftime("%d/%m/%Y")
ICON_FILE = Path("trainIcon.ico")


def logo_pixmap(color: QtGui.QColor = QtGui.QColor("#00ff7f")) -> QtGui.QPixmap:
    """Convert the embedded monochrome logo bitmap into a pixmap."""
    logo_bitmap = QtGui.QBitmap.fromData(QtCore.QSize(64, 64), LOGO_BITMAP, QtGui.QImage.Format_Mono)
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QBrush(color))
    painter.drawPixmap(0, 0, QtGui.QPixmap.fromImage(logo_bitmap.toImage()))
    painter.end()
    return pixmap


def detect_gpus() -> List[str]:
    """Return a list of available GPU strings using nvidia-smi when available."""
    process = QtCore.QProcess()
    process.start("nvidia-smi", ["--query-gpu=index,name", "--format=csv,noheader"])
    if not process.waitForFinished(2000):
        return []
    text = process.readAllStandardOutput().data().decode("utf-8", errors="ignore")

    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            gpus.append(f"GPU {parts[0]}: {parts[1]}")
    return gpus


def human_eta(seconds: float) -> str:
    """Convert seconds into a friendly ETA string."""
    if seconds <= 0:
        return "--:--"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes:02d}m {sec:02d}s"


def read_classes(path: Path) -> List[str]:
    """Read class names from a plain text file (one class per line)."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        if ICON_FILE.exists():
            self.setWindowIcon(QtGui.QIcon(str(ICON_FILE)))

        # Basic menu actions for quit/help.
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("File")
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.qApp.quit)
        help_menu = self.menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        keys_action = help_menu.addAction("Key Bindings")
        keys_action.triggered.connect(self.show_key_bindings)

        self.quit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(QtWidgets.qApp.quit)

        # QProcess is used so we can stream training output live into the UI.
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.training_finished)
        self.process.errorOccurred.connect(self.training_error)
        self.process.started.connect(self.training_started)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Training metadata for ETA/progress estimation.
        self.start_time: Optional[float] = None
        self.total_epochs: Optional[int] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QFormLayout()

        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_button = QtWidgets.QPushButton("Browse")
        self.dataset_button.clicked.connect(self.browse_dataset)
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        dataset_row.addWidget(self.dataset_button)
        form.addRow("Dataset folder:", dataset_row)

        self.classes_edit = QtWidgets.QLineEdit()
        self.classes_button = QtWidgets.QPushButton("Browse")
        self.classes_button.clicked.connect(self.browse_classes)
        classes_row = QtWidgets.QHBoxLayout()
        classes_row.addWidget(self.classes_edit)
        classes_row.addWidget(self.classes_button)
        form.addRow("Classes file:", classes_row)

        # Offer common YOLOv8 model sizes for convenience.
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(
            [
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolov8l.pt",
                "yolov8x.pt",
            ]
        )
        form.addRow("Model version:", self.model_combo)

        # Threads are passed to Ultralytics "workers" to speed up data loading.
        self.thread_spin = QtWidgets.QSpinBox()
        self.thread_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.thread_spin.setValue(min(8, self.thread_spin.maximum()))
        form.addRow("Threads:", self.thread_spin)

        # Allow the user to disable GPU usage quickly if CUDA is misconfigured.
        self.gpu_enable = QtWidgets.QCheckBox("Enable GPU acceleration")
        self.gpu_enable.setChecked(True)
        self.gpu_enable.toggled.connect(self.toggle_gpu)
        form.addRow("GPU:", self.gpu_enable)

        self.gpu_combo = QtWidgets.QComboBox()
        self.refresh_gpus()
        form.addRow("GPU device:", self.gpu_combo)

        layout.addLayout(form)

        self.eta_label = QtWidgets.QLabel("ETA: --:--")
        layout.addWidget(self.eta_label)

        # Progress bar is indeterminate until we parse epoch counts from logs.
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        button_row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        self.resize(720, 640)

    def show_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About",
            f"{WINDOW_TITLE}\nBuild date: {BUILD_DATE}",
        )

    def show_key_bindings(self) -> None:
        html = """
<b>Key Bindings</b>
<pre>
Ctrl+Q        Quit
</pre>
"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Key Bindings")
        msg.setText(html)
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def browse_dataset(self) -> None:
        """Select the dataset folder that contains images and labels."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if directory:
            self.dataset_edit.setText(directory)

    def browse_classes(self) -> None:
        """Select the classes.txt file that lists class names."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select classes.txt", filter="Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.classes_edit.setText(path)

    def refresh_gpus(self) -> None:
        """Populate the GPU dropdown by calling nvidia-smi."""
        self.gpu_combo.clear()
        gpus = detect_gpus()
        if not gpus:
            self.gpu_combo.addItem("CPU only")
        else:
            self.gpu_combo.addItem("Auto (first GPU)")
            for gpu in gpus:
                self.gpu_combo.addItem(gpu)

    def toggle_gpu(self, enabled: bool) -> None:
        self.gpu_combo.setEnabled(enabled)

    def log(self, message: str) -> None:
        """Append a timestamped line to the log view."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def prepare_dataset(self, dataset_dir: Path, classes_path: Path) -> Optional[Path]:
        """Build a temporary YOLO folder structure with train/val splits."""
        images = (
            sorted(dataset_dir.glob("*.jpeg"))
            + sorted(dataset_dir.glob("*.jpg"))
            + sorted(dataset_dir.glob("*.png"))
            + sorted(dataset_dir.glob("*.bmp"))
        )
        labeled = [img for img in images if img.with_suffix(".txt").exists()]
        if not labeled:
            QtWidgets.QMessageBox.warning(self, "No data", "No labeled images were found in that folder.")
            return None

        classes = read_classes(classes_path)
        if not classes:
            QtWidgets.QMessageBox.warning(self, "Missing classes", "The classes file is empty or missing.")
            return None

        # Cache folder keeps training data separate from raw captures.
        cache_dir = dataset_dir / ".yolo_training_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        (cache_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (cache_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (cache_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (cache_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Use a fixed seed so the train/val split stays reproducible.
        rng = random.Random(42)
        rng.shuffle(labeled)
        split_index = max(1, int(len(labeled) * 0.8))
        train_items = labeled[:split_index]
        val_items = labeled[split_index:] or labeled[:1]

        def link_or_copy(src: Path, dst: Path) -> None:
            """Symlink if possible to save space; otherwise copy the file."""
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

        for img in train_items:
            link_or_copy(img, cache_dir / "images" / "train" / img.name)
            link_or_copy(img.with_suffix(".txt"), cache_dir / "labels" / "train" / img.with_suffix(".txt").name)
        for img in val_items:
            link_or_copy(img, cache_dir / "images" / "val" / img.name)
            link_or_copy(img.with_suffix(".txt"), cache_dir / "labels" / "val" / img.with_suffix(".txt").name)

        yaml_lines = [
            f"path: {cache_dir}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        yaml_lines.extend([f"  {idx}: {name}" for idx, name in enumerate(classes)])
        data_yaml = cache_dir / "data.yaml"
        data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
        return data_yaml

    def start_training(self) -> None:
        """Validate inputs, build cache, then run Ultralytics CLI."""
        dataset_text = self.dataset_edit.text().strip()
        classes_text = self.classes_edit.text().strip()
        if not dataset_text or not classes_text:
            QtWidgets.QMessageBox.warning(self, "Missing info", "Please select the dataset and classes.txt.")
            return

        dataset_dir = Path(dataset_text)
        classes_path = Path(classes_text)
        if not dataset_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Missing data", "The dataset folder does not exist.")
            return
        if not classes_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing classes", "The classes file does not exist.")
            return
        data_yaml = self.prepare_dataset(dataset_dir, classes_path)
        if data_yaml is None:
            return

        model = self.model_combo.currentText()
        threads = self.thread_spin.value()
        device = "cpu"
        if self.gpu_enable.isChecked() and self.gpu_combo.currentText() != "CPU only":
            if self.gpu_combo.currentText().startswith("GPU"):
                device = self.gpu_combo.currentText().split(":", 1)[0].split()[-1]
            else:
                device = "0"

        yolo_cli = self._resolve_yolo_cli()
        if yolo_cli is None:
            self.log("Could not find Ultralytics CLI (yolo). Ensure ultralytics is installed in this environment.")
            return

        cmd = [
            yolo_cli,
            "detect",
            "train",
            f"data={data_yaml}",
            f"model={model}",
            f"workers={threads}",
            f"device={device}",
        ]

        # Reset UI state before launching training.
        self.log_view.clear()
        self.log("Starting training...")
        self.log(f"Working dir: {Path.cwd()}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"Dataset: {dataset_dir} | Classes: {classes_path} | Cache: {data_yaml.parent}")
        self.start_time = time.time()
        self.total_epochs = None
        self.progress.setRange(0, 0)
        self.eta_label.setText("ETA: --:--")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start(cmd[0], cmd[1:])
        if not self.process.waitForStarted(3000):
            self.log("Failed to start process (yolo command not found or blocked).")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def stop_training(self) -> None:
        """Stop the training subprocess gracefully, then force kill if needed."""
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.log("Stopping training...")
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.log("Terminate timed out; killing process.")
                self.process.kill()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def handle_stdout(self) -> None:
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        if data:
            self.handle_output(data)

    def handle_stderr(self) -> None:
        data = self.process.readAllStandardError().data().decode("utf-8", errors="ignore")
        if data:
            self.handle_output(data)

    def handle_output(self, text: str) -> None:
        for line in text.splitlines():
            if not line.strip():
                continue
            self.log(line)
            self.update_progress_from_line(line)

    def update_progress_from_line(self, line: str) -> None:
        parts = line.strip().split()
        if not parts:
            return
        for token in parts:
            if "/" in token:
                try:
                    current_str, total_str = token.split("/", 1)
                    current = int(current_str)
                    total = int(total_str)
                except ValueError:
                    continue
                if total <= 0:
                    return
                if self.total_epochs != total:
                    self.total_epochs = total
                    self.progress.setRange(0, total)
                self.progress.setValue(current)
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    if current > 0:
                        eta_seconds = elapsed * (total - current) / current
                        self.eta_label.setText(f"ETA: {human_eta(eta_seconds)}")
                return

    def training_finished(self) -> None:
        self.log(f"Training finished. Exit code: {self.process.exitCode()}")
        if self.process.exitCode() != 0:
            self.log("Training ended with errors. Check above logs for details.")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def training_started(self) -> None:
        self.log("Process started.")

    def training_error(self, error: QtCore.QProcess.ProcessError) -> None:
        self.log(f"Process error: {error.name if hasattr(error, 'name') else error}")

    def _resolve_yolo_cli(self) -> Optional[str]:
        """Find the yolo CLI executable within the current environment."""
        exe_dir = Path(sys.executable).parent
        # Windows venv installs yolo.exe alongside python.exe
        candidates = [
            exe_dir / "yolo.exe",
            exe_dir / "ultralytics.exe",
            "yolo",
        ]
        for cand in candidates:
            if isinstance(cand, Path):
                if cand.exists():
                    return str(cand)
            else:
                return cand  # fallback to PATH lookup
        return None


def create_splash() -> QtWidgets.QSplashScreen:
    splash_pix = QtGui.QPixmap(460, 420)
    splash_pix.fill(QtGui.QColor("#f0f0f0"))
    painter = QtGui.QPainter(splash_pix)
    painter.setPen(QtGui.QColor("#1f1f1f"))
    painter.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
    painter.drawText(splash_pix.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, WINDOW_TITLE)

    logo = logo_pixmap(QtGui.QColor("#000000"))
    scaled_logo = logo.scaled(174, 174, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    logo_x = (splash_pix.width() - scaled_logo.width()) // 2
    logo_y = 100
    painter.drawPixmap(logo_x, logo_y, scaled_logo)

    painter.setFont(QtGui.QFont("Segoe UI", 10))
    text_y = logo_y + scaled_logo.height() + 20
    text_height = max(40, splash_pix.height() - text_y - 20)
    painter.drawText(
        20,
        text_y,
        splash_pix.width() - 40,
        text_height,
        QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
        f"Loading Training Program...\nBuild date: {BUILD_DATE}",
    )
    painter.end()

    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    QtWidgets.QApplication.processEvents()
    return splash


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    splash = create_splash()
    window = TrainingWindow()
    window.show()
    splash.finish(window)

    sys.exit(app.exec_())


LOGO_BITMAP = bytes(
    [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x7F, 0xC0, 0x07, 0x80, 0x00, 0x00,
        0x00, 0x00, 0x60, 0xE0, 0x1F, 0xC0, 0x00, 0x00,
        0x00, 0x00, 0xC0, 0x60, 0x18, 0xE0, 0x00, 0x00,
        0x00, 0x00, 0xC0, 0x70, 0x30, 0x60, 0x00, 0x00,
        0x00, 0x00, 0xC0, 0x3F, 0xB0, 0x7F, 0x00, 0x00,
        0x00, 0x00, 0xC0, 0x03, 0xB0, 0x03, 0x00, 0x00,
        0x00, 0x00, 0xC0, 0x03, 0x38, 0x03, 0x00, 0x00,
        0x00, 0x00, 0xE0, 0x03, 0x1E, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x78, 0x06, 0x0E, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x1C, 0x06, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x0C, 0x0C, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x1C, 0x0C, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x1C, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x18, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x30, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x3F, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x7F, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x00, 0xFF, 0xF3, 0x00, 0x00,
        0x00, 0x00, 0x18, 0x7F, 0xFF, 0xFB, 0x00, 0x00,
        0x00, 0x00, 0x1F, 0xFE, 0x00, 0x0F, 0x00, 0x00,
        0x00, 0x00, 0x1F, 0x00, 0x00, 0x0F, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x1B, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x7B, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xE3, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x03, 0xC3, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x07, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x1E, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x78, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0xE0, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x07, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x1F, 0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]
)


if __name__ == "__main__":
    main()
