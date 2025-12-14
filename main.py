"""Camera capture and bounding-box labeling tool (PyQt + OpenCV).

- Live camera preview in a desktop window.
- Capture a frame (button or `c`) and immediately label it with boxes.
- YOLO-format labels are saved alongside each captured image.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


# Constants
WINDOW_TITLE = "OpenCV Capture and Label"
CAPTURE_DIR = Path("captures")
CLASSES_PATH = Path("classes.txt")
CLASS_COLORS = [
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
TIMER_INTERVAL_MS = 33  # ~30 FPS for the preview timer

# Global variables (none)


def load_classes(path: Path = CLASSES_PATH) -> List[str]:
    """Read class names from classes.txt; fall back to a single 'defect' class."""
    if path.exists():
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return ["defect"]


def save_labels_yolo(image_path: Path, boxes: List[List[int]], image_shape) -> Path:
    """Write YOLO-format labels for the provided boxes next to the image."""
    height, width = image_shape[:2]
    label_path = image_path.with_suffix(".txt")
    lines = []
    for x1, y1, x2, y2, class_id in boxes:
        # Ensure coordinates are ordered and within bounds
        x_min, x_max = sorted((max(0, x1), min(width - 1, x2)))
        y_min, y_max = sorted((max(0, y1), min(height - 1, y2)))
        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w <= 0 or box_h <= 0:
            continue
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        w_norm = box_w / width
        h_norm = box_h / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")
    return label_path


def class_color(class_id: int) -> Tuple[int, int, int]:
    """Pick a color for a class id from a fixed palette."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def draw_text_with_bg(
    img,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Render text with a solid background for readability."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 2, y - text_h - baseline - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def annotate_image(image_path: Path, classes: List[str]) -> bool:
    """Open a simple OpenCV window to draw boxes and save YOLO labels.

    Mouse: click-drag to draw a box.
    Keys:
      - number keys: choose a class id (0-9) for the selected box
      - Enter: apply the chosen class to the selected box
      - left/right arrows: move selection between boxes
      - u / z: undo last box
      - s: save labels and finish
      - q or ESC: cancel labeling

    Returns True if labels were saved, False if the user cancelled.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image for annotation: {image_path}")
        return False

    window_name = f"Annotate: {image_path.name}"
    boxes: List[List[Optional[int]]] = []  # [x1, y1, x2, y2, class_id or None]
    drawing = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None
    selected_index: Optional[int] = None  # which box we are focusing on
    pending_class_choice: Optional[int] = None  # class chosen by number key, applied on Enter
    flash_counter = 0

    img_height, img_width = image.shape[:2]

    def clamp_point(x: int, y: int) -> Tuple[int, int]:
        return max(0, min(x, img_width - 1)), max(0, min(y, img_height - 1))

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001 - OpenCV callback signature
        nonlocal drawing, start_point, current_point, boxes, selected_index, pending_class_choice
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = clamp_point(x, y)
            current_point = start_point
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = clamp_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end_point = clamp_point(x, y)
            if start_point and end_point and start_point != end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), None])
                selected_index = len(boxes) - 1
                pending_class_choice = None
            start_point = None
            current_point = None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        flash_counter += 1
        flash_on = (flash_counter // 15) % 2 == 0  # toggle roughly every 0.5s at ~30 fps

        # If the user manually closed the annotation window, exit gracefully.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            return False

        # Draw boxes and the in-progress rectangle over a copy of the image.
        canvas = image.copy()
        for idx, (x1, y1, x2, y2, cls) in enumerate(boxes):
            # Preview pending class on the selected box before applying.
            display_cls: Optional[int] = pending_class_choice if (idx == selected_index and pending_class_choice is not None) else cls
            base_color = class_color(display_cls) if display_cls is not None else (0, 165, 255)
            color = (255, 255, 255) if (idx == selected_index and flash_on) else base_color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if display_cls is not None:
                label = (classes[display_cls] if display_cls < len(classes) else str(display_cls)).upper()
                suffix = " (PENDING)" if (idx == selected_index and pending_class_choice is not None and cls is None) else ""
                draw_text_with_bg(
                    canvas,
                    f"{display_cls}: {label}{suffix}",
                    (x1, max(15, y1 - 5)),
                    font_scale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                )

        if drawing and start_point and current_point:
            cv2.rectangle(canvas, start_point, current_point, (0, 165, 255), 1)

        # On-screen instructions for quick reference.
        top_line = "Drag: draw box | 0-9: pick class | enter: apply to selected"
        top_stats = f"Boxes: {len(boxes)} (pending labels: {sum(1 for _, _, _, _, cls in boxes if cls is None)})"
        bottom_instructions = [
            "<-/->: select box | u/z: undo | s: save | q/esc: cancel",
        ]
        top_line = top_line.upper()
        top_stats = top_stats.upper()
        bottom_instructions = [line.upper() for line in bottom_instructions]

        # Draw the primary instruction and stats at the top.
        draw_text_with_bg(
            canvas,
            top_line,
            (10, 24),
            font_scale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )
        draw_text_with_bg(
            canvas,
            top_stats,
            (10, 48),
            font_scale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )

        # Draw secondary instruction lines anchored near the bottom.
        line_height = 24
        y_offset = img_height - 10 - line_height * len(bottom_instructions)
        for line in bottom_instructions:
            y_offset += line_height
            draw_text_with_bg(
                canvas,
                line,
                (10, y_offset),
                font_scale=0.5,
                color=(0, 255, 0),  # green text for visibility
                thickness=1,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            if any(cls is None for *_, cls in boxes):
                print("Label all boxes with number keys before saving.")
                continue
            save_labels_yolo(image_path, boxes, image.shape)
            cv2.destroyWindow(window_name)
            return True
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return False
        if key in (ord("u"), ord("z")) and boxes:
            boxes.pop()
            selected_index = len(boxes) - 1 if boxes else None
            pending_class_choice = None
        if ord("0") <= key <= ord("9"):
            class_id = key - ord("0")
            if boxes:
                if selected_index is None:
                    selected_index = len(boxes) - 1
                pending_class_choice = class_id
                print(f"Class {class_id} selected. Press Enter to apply.")
            else:
                print("No box to label. Draw a box first.")
        if key in (13, 10):  # Enter/Return
            if selected_index is None or not boxes:
                print("Select a box first.")
            elif pending_class_choice is None:
                print("Pick a class (0-9) before applying.")
            else:
                boxes[selected_index][4] = pending_class_choice
                pending_class_choice = None
                selected_index = None  # deselect to stop flashing after applying
        # Arrow keys to move selection; include common OpenCV keycodes.
        LEFT_KEYS = {81, 2424832}
        RIGHT_KEYS = {83, 2555904}
        if key in LEFT_KEYS and boxes:
            selected_index = (len(boxes) - 1) if selected_index is None else (selected_index - 1) % len(boxes)
            pending_class_choice = None
        if key in RIGHT_KEYS and boxes:
            selected_index = 0 if selected_index is None else (selected_index + 1) % len(boxes)
            pending_class_choice = None


def open_camera(index: int, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    """Create and configure an OpenCV VideoCapture for the chosen camera index."""
    cap = cv2.VideoCapture(index)

    # Let callers request a particular resolution. If the camera cannot satisfy the
    # request, it will keep its closest supported resolution.
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def enumerate_cameras(max_index: int = 5) -> List[int]:
    """Probe a handful of indices to find available cameras."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found


class CameraWindow(QtWidgets.QWidget):
    """Simple window that shows live frames from a camera using Qt widgets."""

    def __init__(self, camera_index: int, width: Optional[int], height: Optional[int]):
        super().__init__()

        self.classes = load_classes()
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.cap = open_camera(camera_index, width, height)
        self.current_frame = None  # Most recent BGR frame from the camera

        # Top navigation bar with menu + camera selector.
        self.menu_bar = QtWidgets.QMenuBar()
        file_menu = self.menu_bar.addMenu("File")
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.qApp.quit)

        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.populate_camera_selector()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)

        nav_bar = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        nav_layout.addWidget(self.menu_bar)
        nav_layout.addStretch()
        nav_layout.addWidget(QtWidgets.QLabel("Camera:"))
        nav_layout.addWidget(self.camera_selector)

        # Basic UI: video preview, capture button, status text.
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Connecting to camera...")

        self.capture_button = QtWidgets.QPushButton("Capture (C)")
        self.capture_button.clicked.connect(self.capture_frame)
        self.status_label = QtWidgets.QLabel("Ready")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(nav_bar)
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.setWindowTitle(WINDOW_TITLE)

        # A Qt timer calls `update_frame` ~30 times per second to pull fresh frames
        # from OpenCV without blocking the UI event loop.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(TIMER_INTERVAL_MS)

        if not self.cap.isOpened():
            self.video_label.setText(f"Unable to open camera index {camera_index}.")

    def populate_camera_selector(self) -> None:
        """Fill the dropdown with discovered camera indices, keeping the current one if missing."""
        available = enumerate_cameras()
        if not available:
            available = [self.camera_index]

        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for idx in available:
            self.camera_selector.addItem(f"Camera {idx}", idx)

        if self.camera_selector.findData(self.camera_index) == -1:
            self.camera_selector.addItem(f"Camera {self.camera_index}", self.camera_index)

        current_idx = self.camera_selector.findData(self.camera_index)
        if current_idx >= 0:
            self.camera_selector.setCurrentIndex(current_idx)
        self.camera_selector.blockSignals(False)

    def change_camera(self, combo_index: int) -> None:
        """Switch to the camera selected in the dropdown."""
        new_index = self.camera_selector.itemData(combo_index)
        if new_index is None or new_index == self.camera_index:
            return

        self.timer.stop()
        new_cap = open_camera(new_index, self.requested_width, self.requested_height)
        if not new_cap.isOpened():
            self.status_label.setText(f"Unable to open camera {new_index}.")
            new_cap.release()
            self.timer.start(TIMER_INTERVAL_MS)
            self.populate_camera_selector()
            return

        if self.cap.isOpened():
            self.cap.release()
        self.cap = new_cap
        self.camera_index = new_index
        self.status_label.setText(f"Switched to camera {new_index}.")
        self.timer.start(TIMER_INTERVAL_MS)

    def update_frame(self) -> None:
        """Grab a frame from OpenCV, convert it, and display it in the QLabel."""
        if not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok:
            self.video_label.setText("Failed to read frame from camera.")
            self.timer.stop()
            return

        self.current_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width

        q_image = QtGui.QImage(
            frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)

        self.video_label.setPixmap(pixmap)
        self.video_label.setFixedSize(pixmap.size())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        """Release the camera when the window closes to free the device."""
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        # Ensure any OpenCV windows are cleaned up to avoid crashes on exit.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        super().closeEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        """Allow pressing 'c' to capture a frame."""
        if event.key() in (QtCore.Qt.Key_C,):
            self.capture_frame()
        else:
            super().keyPressEvent(event)

    def capture_frame(self) -> None:
        """Save the most recent frame to disk, then force labeling before continuing."""
        if self.current_frame is None:
            self.status_label.setText("No frame available yet.")
            return

        # Pause the live feed while we handle capture + labeling, so nothing runs in the background.
        self.timer.stop()
        self.capture_button.setEnabled(False)

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = CAPTURE_DIR / f"capture_{timestamp}.jpg"

        success = cv2.imwrite(str(image_path), self.current_frame)
        if not success:
            self.status_label.setText("Failed to save image.")
            self.capture_button.setEnabled(True)
            self.timer.start(TIMER_INTERVAL_MS)
            return

        self.status_label.setText(f"Saved {image_path.name}. Please label it.")

        labeled = annotate_image(image_path, self.classes)
        if labeled:
            self.status_label.setText(f"Labeled {image_path.name}.")
        else:
            # If labeling was cancelled, remove the capture so nothing remains unlabeled.
            try:
                image_path.unlink(missing_ok=True)
                image_path.with_suffix(".txt").unlink(missing_ok=True)
            except OSError:
                pass
            self.status_label.setText("Labeling cancelled. Capture discarded.")

        # Resume the live feed and allow the next capture.
        self.capture_button.setEnabled(True)
        self.timer.start(TIMER_INTERVAL_MS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display a live camera feed in a PyQt window.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Optional frame width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Optional frame height in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = CameraWindow(args.camera, args.width, args.height)
    window.show()

    # Start the Qt event loop; this keeps the window responsive until closed.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
