"""OpenCV Capture and Label (PyQt).

This desktop tool provides:
- Live camera preview with capture + mandatory labeling flow.
- Keyboard-driven annotator (boxes, class selection, pan/zoom) with null/discard paths.
- Camera inspection tools (class/color editor, camera settings dialog).
- Dataset inspection mode to browse/edit/delete labeled images.
- Portable packaging via PyInstaller (splash, icons included).
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


# Constants and configuration
WINDOW_TITLE = "Capture and Label"
CAPTURE_DIR = Path("captures")
NULL_DIR = CAPTURE_DIR / "null"
CLASSES_PATH = Path("classes.txt")
CLASS_COLORS_PATH = Path("class_colors.json")
ICON_FILE = Path("captureIcon.ico")
if getattr(sys, "frozen", False):
    build_source = Path(sys.executable)
else:
    build_source = Path(__file__)
try:
    BUILD_DATE = datetime.fromtimestamp(build_source.stat().st_mtime).strftime("%d/%m/%Y")
except OSError:
    BUILD_DATE = datetime.now().strftime("%d/%m/%Y")
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
TIMER_INTERVAL_MS = 33  # ~30 FPS for the preview timer
LOGO_WIDTH = 64
LOGO_HEIGHT = 64
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
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]
)

# Global variables (none)


def load_classes(path: Path = CLASSES_PATH) -> List[str]:
    """Read class names from classes.txt; fall back to a single 'defect' class."""
    if path.exists():
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return ["defect"]


def save_classes(classes: List[str], path: Path = CLASSES_PATH) -> None:
    """Persist class names to disk, one per line."""
    path.write_text("\n".join(classes), encoding="utf-8")


def load_class_colors(path: Path = CLASS_COLORS_PATH) -> List[Tuple[int, int, int]]:
    """Load class color palette from JSON; fall back to defaults on errors."""
    if not path.exists():
        return list(DEFAULT_CLASS_COLORS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        colors: List[Tuple[int, int, int]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                r, g, b = (int(max(0, min(255, v))) for v in item)
                colors.append((r, g, b))
        return colors or list(DEFAULT_CLASS_COLORS)
    except (json.JSONDecodeError, OSError, ValueError):
        return list(DEFAULT_CLASS_COLORS)


def save_class_colors(colors: List[Tuple[int, int, int]], path: Path = CLASS_COLORS_PATH) -> None:
    """Persist class colors to disk as a JSON array of RGB triples."""
    path.write_text(json.dumps(colors), encoding="utf-8")


def class_color(class_id: int, palette: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Pick a color for a class id from the provided palette."""
    if not palette:
        return 0, 255, 0
    return palette[class_id % len(palette)]


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


def load_labels_yolo(label_path: Path, image_shape) -> List[List[int]]:
    """Load YOLO labels and convert to pixel boxes."""
    # Accept either a shape tuple or a numpy image array.
    if hasattr(image_shape, "shape"):
        height, width = image_shape.shape[:2]
    else:
        height, width = image_shape[:2]
    boxes: List[List[int]] = []
    if not label_path.exists():
        return boxes
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x_c, y_c, w_norm, h_norm = map(float, parts[1:5])
            w_px = w_norm * width
            h_px = h_norm * height
            x1 = (x_c * width) - w_px / 2
            y1 = (y_c * height) - h_px / 2
            x2 = x1 + w_px
            y2 = y1 + h_px
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(width - 1, int(round(x2))), min(height - 1, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2, cls])
    except OSError:
        pass
    return boxes


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


def logo_pixmap(color: QtGui.QColor = QtGui.QColor("#00ff7f")) -> QtGui.QPixmap:
    """Convert the embedded monochrome logo bitmap into a pixmap."""
    img = QtGui.QImage(LOGO_WIDTH, LOGO_HEIGHT, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    for y in range(LOGO_HEIGHT):
        for x in range(LOGO_WIDTH):
            byte_index = (y * LOGO_WIDTH + x) // 8
            bit_index = 7 - (x % 8)
            if byte_index < len(LOGO_BITMAP) and ((LOGO_BITMAP[byte_index] >> bit_index) & 1):
                img.setPixelColor(x, y, color)
    return QtGui.QPixmap.fromImage(img)


def load_app_icon() -> Optional[QtGui.QIcon]:
    """Load the app icon (programLogo.ico) with PyInstaller support."""
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


def annotate_image(
    image_path: Path,
    classes: List[str],
    class_colors: List[Tuple[int, int, int]],
    initial_boxes: Optional[List[List[int]]] = None,
) -> str:
    """Open a simple OpenCV window to draw boxes and save YOLO labels.

    Mouse: click-drag to draw a box.
    Keys:
      - number keys: choose a class id (0-9) for the selected box
      - Enter: apply the chosen class to the selected box
      - left/right arrows: move selection between boxes
      - u / z: undo last box
      - s: save labels and finish
      - n: mark as null (no defects) and move capture to null folder (only when no boxes)
      - q or ESC: cancel labeling

    Returns one of: "saved", "null", "cancel".
    """

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image for annotation: {image_path}")
        return "cancel"

    # Window title helps the user confirm which image is being labeled.
    window_name = f"Annotate: {image_path.name}"
    boxes: List[List[Optional[int]]] = []  # [x1, y1, x2, y2, class_id or None]
    if initial_boxes:
        for b in initial_boxes:
            if len(b) == 5:
                boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4])])
    drawing = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None
    selected_index: Optional[int] = None  # which box we are focusing on
    pending_class_choice: Optional[int] = None  # class chosen by number key, applied on Enter
    flash_counter = 0

    img_height, img_width = image.shape[:2]

    def safe_destroy_window(name: str) -> None:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass

    zoom_factor = 1.0
    pan_x = 0.0
    pan_y = 0.0
    pan_active = False
    pan_start_view: Optional[Tuple[int, int]] = None
    # Store view state as (view width, view height, center x, center y).
    current_view_params: Tuple[float, float, float, float] = (
        float(img_width),
        float(img_height),
        float(img_width) / 2.0,
        float(img_height) / 2.0,
    )

    # Clamp a point so we never access outside the image.
    def clamp_point(x: float, y: float) -> Tuple[int, int]:
        return int(max(0, min(round(x), img_width - 1))), int(max(0, min(round(y), img_height - 1)))

    # Convert view coordinates into image coordinates.
    def view_to_image_coords(vx: int, vy: int) -> Tuple[int, int]:
        roi_w, roi_h, cx, cy = current_view_params
        img_x = (vx / img_width) * roi_w + (cx - roi_w / 2.0)
        img_y = (vy / img_height) * roi_h + (cy - roi_h / 2.0)
        return clamp_point(img_x, img_y)

    # Zoom while keeping the mouse position anchored.
    def adjust_zoom(direction: int, vx: int, vy: int) -> None:
        nonlocal zoom_factor, pan_x, pan_y
        if direction == 0:
            return
        before_zoom = zoom_factor
        step = 1.1 if direction > 0 else (1 / 1.1)
        zoom_factor = max(0.5, min(4.0, zoom_factor * step))
        if abs(zoom_factor - before_zoom) < 1e-6:
            return
        # Keep the point under the cursor stable after zoom.
        target_x, target_y = view_to_image_coords(vx, vy)
        fx = vx / img_width
        fy = vy / img_height
        new_roi_w = img_width / zoom_factor
        new_roi_h = img_height / zoom_factor
        new_cx = target_x - fx * new_roi_w + new_roi_w / 2.0
        new_cy = target_y - fy * new_roi_h + new_roi_h / 2.0
        pan_x = new_cx - (img_width / 2.0)
        pan_y = new_cy - (img_height / 2.0)

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001 - OpenCV callback signature
        nonlocal drawing, start_point, current_point, boxes, selected_index, pending_class_choice, pan_active, pan_start_view, pan_x, pan_y
        # Left mouse button starts drawing a new box.
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = view_to_image_coords(x, y)
            current_point = start_point
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = view_to_image_coords(x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end_point = view_to_image_coords(x, y)
            if start_point and end_point and start_point != end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), None])
                selected_index = len(boxes) - 1
                pending_class_choice = None
            start_point = None
            current_point = None
        # Middle mouse button pans the view.
        elif event == cv2.EVENT_MBUTTONDOWN:
            pan_active = True
            pan_start_view = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            pan_active = False
            pan_start_view = None
        elif event == cv2.EVENT_MOUSEMOVE and pan_active and pan_start_view:
            # Convert drag delta from view to image space.
            dx_view = x - pan_start_view[0]
            dy_view = y - pan_start_view[1]
            roi_w, roi_h, _, _ = current_view_params
            pan_x -= (dx_view * roi_w) / img_width
            pan_y -= (dy_view * roi_h) / img_height
            pan_start_view = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = 1 if flags > 0 else -1
            adjust_zoom(direction, x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        flash_counter += 1
        flash_on = (flash_counter // 15) % 2 == 0  # toggle roughly every 0.5s at ~30 fps

        # If the user manually closed the annotation window, exit gracefully.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            safe_destroy_window(window_name)
            return "cancel"

        # Apply zoom/pan to the base image; draw overlays after scaling.
        roi_w = int(round(img_width / zoom_factor))
        roi_h = int(round(img_height / zoom_factor))
        roi_w = max(1, min(img_width, roi_w))
        roi_h = max(1, min(img_height, roi_h))
        cx = (img_width / 2.0) + pan_x
        cy = (img_height / 2.0) + pan_y
        # Clamp center to keep ROI inside image bounds.
        half_w = roi_w / 2.0
        half_h = roi_h / 2.0
        cx = max(half_w, min(img_width - half_w, cx))
        cy = max(half_h, min(img_height - half_h, cy))
        current_view_params = (float(roi_w), float(roi_h), cx, cy)
        roi = cv2.getRectSubPix(image, (roi_w, roi_h), (cx, cy))
        view = cv2.resize(roi, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        # Convert image coordinates to view coordinates for drawing overlays.
        def img_to_view(ix: int, iy: int) -> Tuple[int, int]:
            vx = int(round((ix - (cx - roi_w / 2.0)) * (img_width / roi_w)))
            vy = int(round((iy - (cy - roi_h / 2.0)) * (img_height / roi_h)))
            vx = max(0, min(img_width - 1, vx))
            vy = max(0, min(img_height - 1, vy))
            return vx, vy

        # Draw boxes and the in-progress rectangle after scaling.
        for idx, (x1, y1, x2, y2, cls) in enumerate(boxes):
            display_cls: Optional[int] = pending_class_choice if (idx == selected_index and pending_class_choice is not None) else cls
            base_color = class_color(display_cls, class_colors) if display_cls is not None else (0, 165, 255)
            color = (255, 255, 255) if (idx == selected_index and flash_on) else base_color
            v1 = img_to_view(x1, y1)
            v2 = img_to_view(x2, y2)
            cv2.rectangle(view, v1, v2, color, 2)
            if display_cls is not None:
                label = (classes[display_cls] if display_cls < len(classes) else str(display_cls)).upper()
                suffix = " (PENDING)" if (idx == selected_index and pending_class_choice is not None and cls is None) else ""
                draw_text_with_bg(
                    view,
                    f"{display_cls}: {label}{suffix}",
                    (v1[0], max(15, v1[1] - 5)),
                    font_scale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                )

        if drawing and start_point and current_point:
            v_start = img_to_view(*start_point)
            v_cur = img_to_view(*current_point)
            cv2.rectangle(view, v_start, v_cur, (0, 165, 255), 1)

        # On-screen instructions for quick reference (drawn after zoom so they stay static).
        top_line = "Drag: draw box | 0-9: pick class | enter: apply to selected"
        top_stats = f"Boxes: {len(boxes)} (pending labels: {sum(1 for _, _, _, _, cls in boxes if cls is None)})"
        bottom_instructions = [
            "<-/->: select box | u/z: undo | s: save | n: null | q/esc: cancel",
            "scroll: zoom | mmb drag: pan",
        ]
        top_line = top_line.upper()
        top_stats = top_stats.upper()
        bottom_instructions = [line.upper() for line in bottom_instructions]

        draw_text_with_bg(
            view,
            top_line,
            (10, 24),
            font_scale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )
        draw_text_with_bg(
            view,
            top_stats,
            (10, 48),
            font_scale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )
        line_height = 24
        y_offset = img_height - 10 - line_height * len(bottom_instructions)
        for line in bottom_instructions:
            y_offset += line_height
            draw_text_with_bg(
                view,
                line,
                (10, y_offset),
                font_scale=0.5,
                color=(0, 255, 0),
                thickness=1,
            )

        cv2.imshow(window_name, view)
        # waitKeyEx preserves extended key codes (e.g., arrow keys).
        key = cv2.waitKeyEx(30)

        if key == ord("s"):
            if any(cls is None for *_, cls in boxes):
                print("Label all boxes with number keys before saving.")
                continue
            save_labels_yolo(image_path, boxes, image.shape)
            safe_destroy_window(window_name)
            return "saved"
        if key in (ord("q"), 27):
            safe_destroy_window(window_name)
            return "cancel"
        if key == ord("n"):
            if boxes:
                print("Remove boxes or undo before marking null.")
                continue
            safe_destroy_window(window_name)
            return "null"
        if key in (ord("u"), ord("z")) and boxes:
            boxes.pop()
            selected_index = len(boxes) - 1 if boxes else None
            pending_class_choice = None
        if ord("0") <= key <= ord("9"):
            class_id = key - ord("0")
            if class_id >= len(classes):
                print(f"Class {class_id} not defined. Only 0-{len(classes) - 1} available.")
            elif boxes:
                if selected_index is None:
                    selected_index = len(boxes) - 1
                pending_class_choice = class_id
                print(f"Class {class_id} selected. Press Enter to apply.")
            else:
                print("No box to label. Draw a box first.")
        UP_KEYS = {82, 2490368}
        DOWN_KEYS = {84, 2621440}
        if key in UP_KEYS.union(DOWN_KEYS):
            if not classes:
                print("No classes available.")
            elif boxes:
                if selected_index is None:
                    selected_index = len(boxes) - 1
                current_cls = pending_class_choice
                if current_cls is None:
                    current_cls = boxes[selected_index][4] if boxes[selected_index][4] is not None else 0
                delta = 1 if key in UP_KEYS else -1
                new_cls = (current_cls + delta) % len(classes)
                pending_class_choice = new_cls
                print(f"Class {new_cls} selected with arrow key. Press Enter to apply.")
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


class ClassSettingsDialog(QtWidgets.QDialog):
    """Dialog to edit class names and colors."""

    def __init__(self, classes: List[str], colors: List[Tuple[int, int, int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Settings")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.resize(400, 500)
        self._base_classes = list(classes)
        self._base_colors = list(colors) if colors else list(DEFAULT_CLASS_COLORS)
        self.rows: List[dict] = []

        count_label = QtWidgets.QLabel("Number of classes:")
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(50)
        self.count_spin.setValue(max(1, len(classes)))
        self.count_spin.valueChanged.connect(self._rebuild_rows)

        count_layout = QtWidgets.QHBoxLayout()
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_spin)
        count_layout.addStretch()

        self.rows_container = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rows_container)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(count_layout)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Recreate rows to match the chosen class count."""
        # Preserve current edits before rebuilding so names/colors are not lost.
        if self.rows:
            current_names: List[str] = []
            current_colors: List[Tuple[int, int, int]] = []
            for idx, row in enumerate(self.rows):
                text = row["name"].text().strip()
                current_names.append(text if text else f"class_{idx}")
                current_colors.append(row["color"])
            self._base_classes = current_names
            self._base_colors = current_colors

        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.rows = []

        count = self.count_spin.value()
        for idx in range(count):
            name = self._base_classes[idx] if idx < len(self._base_classes) else f"class_{idx}"
            color = self._base_colors[idx] if idx < len(self._base_colors) else DEFAULT_CLASS_COLORS[idx % len(DEFAULT_CLASS_COLORS)]

            name_edit = QtWidgets.QLineEdit(name)
            color_button = QtWidgets.QPushButton()
            color_button.setFixedWidth(60)
            self._set_button_color(color_button, color)

            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(QtWidgets.QLabel(f"{idx}:"))
            row_layout.addWidget(name_edit, stretch=1)
            row_layout.addWidget(color_button)

            self.rows_layout.addWidget(row_widget)

            row_data = {"name": name_edit, "color": color, "button": color_button}
            color_button.clicked.connect(lambda _, r=row_data: self._pick_color(r))
            self.rows.append(row_data)

        self.rows_layout.addStretch()

    def _pick_color(self, row: dict) -> None:
        """Open a color dialog and update the stored color."""
        r, g, b = row["color"]
        current = QtGui.QColor(r, g, b)
        chosen = QtWidgets.QColorDialog.getColor(current, self, "Choose Color")
        if chosen.isValid():
            row["color"] = (chosen.red(), chosen.green(), chosen.blue())
            self._set_button_color(row["button"], row["color"])

    @staticmethod
    def _set_button_color(button: QtWidgets.QPushButton, color: Tuple[int, int, int]) -> None:
        r, g, b = color
        button.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

    def get_results(self) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        """Return sanitized names and colors from the form."""
        names: List[str] = []
        colors: List[Tuple[int, int, int]] = []
        for idx, row in enumerate(self.rows):
            raw_name = row["name"].text().strip()
            names.append(raw_name if raw_name else f"class_{idx}")
            colors.append(row["color"])
        return names, colors



class CameraWindow(QtWidgets.QWidget):
    """Simple window that shows live frames from a camera using Qt widgets."""

    def __init__(self, camera_index: int, width: Optional[int], height: Optional[int]):
        super().__init__()

        self.classes = load_classes()
        self.class_colors = load_class_colors()
        self._sync_palette_with_classes()
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.cap = open_camera(camera_index, width, height)
        self.current_frame = None  # Most recent BGR frame from the camera
        self.inspect_mode = False
        self.inspect_files: List[Path] = []
        self.inspect_index = 0

        # Top navigation bar with menu + camera selector.
        self.menu_bar = QtWidgets.QMenuBar()
        self.menu_bar.setStyleSheet(
            "QMenuBar { background: transparent; }"
            "QMenuBar::item { background: transparent; }"
            "QMenu { background: #f0f0f0; color: black; }"
            "QMenu::item:selected { background: #d0d0d0; }"
        )
        file_menu = self.menu_bar.addMenu("File")
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.qApp.quit)
        inspect_file_action = file_menu.addAction("Inspect File...")
        inspect_file_action.triggered.connect(self.inspect_file)
        inspect_folder_action = file_menu.addAction("Inspect Folder...")
        inspect_folder_action.triggered.connect(self.inspect_folder)
        settings_menu = self.menu_bar.addMenu("Settings")
        edit_classes_action = settings_menu.addAction("Edit Classes && Colors")
        edit_classes_action.triggered.connect(self.open_class_settings)
        edit_camera_action = settings_menu.addAction("Edit Camera Settings")
        edit_camera_action.triggered.connect(self.open_camera_settings)
        help_menu = self.menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        keys_action = help_menu.addAction("Key Bindings")
        keys_action.triggered.connect(self.show_key_bindings)

        self.camera_selector = QtWidgets.QComboBox()
        self.camera_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.camera_selector.setMinimumWidth(120)
        self.camera_selector.setFixedHeight(self.menu_bar.sizeHint().height())
        self.camera_selector.setStyleSheet(
            "QComboBox { background: #f0f0f0; color: black; }"
            "QComboBox QAbstractItemView { background: #f0f0f0; color: black; }"
        )
        self.populate_camera_selector()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)

        nav_bar = QtWidgets.QWidget()
        nav_bar.setAutoFillBackground(False)
        nav_bar.setStyleSheet("background: transparent;")
        nav_layout = QtWidgets.QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        nav_layout.addWidget(self.menu_bar)
        nav_layout.addStretch()
        camera_label = QtWidgets.QLabel("Camera:")
        camera_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        nav_layout.addWidget(camera_label)
        nav_layout.addWidget(self.camera_selector)

        # Basic UI: video preview, capture button, status text.
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_label.setText("Connecting to camera...")
        self._last_pixmap: Optional[QtGui.QPixmap] = None

        self.capture_button = QtWidgets.QPushButton("Capture (C)")
        self.capture_button.clicked.connect(self.capture_frame)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        # Inspect action buttons.
        self.inspect_buttons = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(self.inspect_buttons)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        self.inspect_delete_btn = QtWidgets.QPushButton("Delete (Del)")
        self.inspect_delete_btn.clicked.connect(self._delete_current_inspect)
        self.inspect_edit_btn = QtWidgets.QPushButton("Edit (E)")
        self.inspect_edit_btn.clicked.connect(self._edit_current_inspect)
        self.inspect_delete_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.inspect_edit_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        btn_layout.addWidget(self.inspect_delete_btn, 1)
        btn_layout.addWidget(self.inspect_edit_btn, 1)
        self.inspect_buttons.setEnabled(False)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(nav_bar)
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.inspect_buttons)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.setWindowTitle(WINDOW_TITLE)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Global shortcuts to ensure key presses work regardless of focus.
        self.capture_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("C"), self)
        self.capture_shortcut.activated.connect(self.capture_frame)
        self.quit_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self)
        self.quit_shortcut.activated.connect(self.close)

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

    def _sync_palette_with_classes(self) -> None:
        """Ensure we have a color per class (pad or trim)."""
        needed = len(self.classes)
        if needed <= 0:
            self.classes = ["defect"]
            needed = 1
        if len(self.class_colors) < needed:
            for i in range(needed - len(self.class_colors)):
                self.class_colors.append(
                    DEFAULT_CLASS_COLORS[(len(self.class_colors) + i) % len(DEFAULT_CLASS_COLORS)]
                )
        elif len(self.class_colors) > needed:
            self.class_colors = self.class_colors[:needed]

    def open_class_settings(self) -> None:
        """Open dialog to edit class names and colors."""
        dialog = ClassSettingsDialog(self.classes, self.class_colors, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_classes, new_colors = dialog.get_results()
            self.classes = new_classes
            self.class_colors = new_colors
            self._sync_palette_with_classes()
            save_classes(self.classes)
            save_class_colors(self.class_colors)
            self.status_label.setText("Updated class settings.")

    def open_camera_settings(self) -> None:
        """Open dialog to tweak camera properties (exposure, white balance, gain)."""
        if not self.cap or not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Camera", "Camera is not open.")
            return
        dialog = CameraSettingsDialog(self.cap, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            dialog.apply_settings()
            self.status_label.setText("Updated camera settings.")

    def show_about(self) -> None:
        """Display build information."""
        QtWidgets.QMessageBox.information(
            self,
            "About",
            f"{WINDOW_TITLE}\nBuild date: {BUILD_DATE}"
        )

    def show_key_bindings(self) -> None:
        """Display a list of key bindings."""
        # Use simple HTML with a monospaced <pre> block for aligned columns.
        html = """
<b>Main Window</b>
<pre>
C             Capture frame
Q             Quit
LEFT/RIGHT    Prev/next (inspect mode)
DELETE        Delete current file (inspect mode)
E             Edit current file (inspect mode)
ESC           Exit inspect mode
</pre>
<b>Labeler Window</b>
<pre>
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
</pre>
"""
        QtWidgets.QMessageBox.information(self, "Key Bindings", html)

    def inspect_file(self) -> None:
        """Inspect a single image file with its labels."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(Path.cwd()),
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self.enter_inspect_mode([path], 0)

    def inspect_folder(self) -> None:
        """Inspect a folder of images; navigate with arrow keys."""
        dir_str = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", str(Path.cwd()))
        if not dir_str:
            return
        folder = Path(dir_str)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])
        if not files:
            QtWidgets.QMessageBox.information(self, "Inspect Folder", "No images found in this folder.")
            return
        self.enter_inspect_mode(files, 0)

    def enter_inspect_mode(self, files: List[Path], start_index: int) -> None:
        """Start inspection mode with provided file list."""
        if not files:
            return
        self.inspect_files = files
        self.inspect_index = max(0, min(start_index, len(files) - 1))
        self.inspect_mode = True
        self.timer.stop()
        self.capture_button.setEnabled(False)
        self.inspect_buttons.setEnabled(True)
        self._display_inspect_image()

    def exit_inspect_mode(self) -> None:
        """Exit inspection mode and resume live camera."""
        if not self.inspect_mode:
            return
        self.inspect_mode = False
        self.inspect_files = []
        self.inspect_index = 0
        self.capture_button.setEnabled(True)
        self.inspect_buttons.setEnabled(False)
        self.status_label.setText("Inspect mode exited. Resuming camera.")
        self.timer.start(TIMER_INTERVAL_MS)

    def _display_inspect_image(self) -> None:
        """Load and display the current inspect image with its labels."""
        if not self.inspect_files:
            return
        image_path = self.inspect_files[self.inspect_index]
        image = cv2.imread(str(image_path))
        if image is None:
            self.status_label.setText(f"Could not open {image_path.name}")
            return

        label_path = image_path.with_suffix(".txt")
        boxes: List[List[int]] = load_labels_yolo(label_path, image)

        # Draw boxes
        for x1, y1, x2, y2, cls in boxes:
            color = class_color(cls, self.class_colors)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = self.classes[cls] if 0 <= cls < len(self.classes) else str(cls)
            draw_text_with_bg(image, f"{cls}: {label}", (x1, max(15, y1 - 5)), font_scale=0.5, color=(0, 255, 0), thickness=1)

        # Overlay inspect mode key bindings.
        overlay_lines = [
            "INSPECT MODE:",
            "LEFT/RIGHT: NAV | DELETE: DELETE FILE | E: EDIT | ESC: EXIT",
        ]
        y_pos = 24
        for line in overlay_lines:
            draw_text_with_bg(
                image,
                line,
                (10, y_pos),
                font_scale=0.6,
                color=(0, 255, 0),
                thickness=1,
            )
            y_pos += 24

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self._set_display_pixmap(pixmap)

        self.status_label.setText(
            f"Inspecting {self.inspect_index + 1}/{len(self.inspect_files)}: {image_path.name} (boxes: {len(boxes)})"
        )

    def _delete_current_inspect(self) -> None:
        """Delete current inspected image and its label; advance or exit."""
        if not self.inspect_files:
            return
        image_path = self.inspect_files[self.inspect_index]
        label_path = image_path.with_suffix(".txt")
        try:
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
        except OSError as exc:
            self.status_label.setText(f"Failed to delete: {exc}")
            return

        del self.inspect_files[self.inspect_index]
        if not self.inspect_files:
            self.status_label.setText("Deleted file. No more items; exiting inspect mode.")
            self.exit_inspect_mode()
            return
        if self.inspect_index >= len(self.inspect_files):
            self.inspect_index = len(self.inspect_files) - 1
        self._display_inspect_image()

    def _edit_current_inspect(self) -> None:
        """Open current inspected image in the labeler UI."""
        if not self.inspect_files:
            return
        image_path = self.inspect_files[self.inspect_index]
        image = cv2.imread(str(image_path))
        if image is None:
            self.status_label.setText(f"Could not open {image_path.name}")
            return
        initial_boxes = load_labels_yolo(image_path.with_suffix(".txt"), image)

        # Temporarily pause timer and edit.
        self.timer.stop()
        result = annotate_image(image_path, self.classes, self.class_colors, initial_boxes=initial_boxes)

        if result == "saved":
            self.status_label.setText(f"Re-labeled {image_path.name}.")
        elif result == "null":
            NULL_DIR.mkdir(parents=True, exist_ok=True)
            dest = NULL_DIR / image_path.name
            try:
                image_path.replace(dest)
                image_path.with_suffix(".txt").unlink(missing_ok=True)
                # Remove from inspect list and refresh.
                del self.inspect_files[self.inspect_index]
                if not self.inspect_files:
                    self.status_label.setText("Marked null and exiting inspect mode.")
                    self.exit_inspect_mode()
                    return
                if self.inspect_index >= len(self.inspect_files):
                    self.inspect_index = len(self.inspect_files) - 1
                self.status_label.setText(f"Marked null -> {dest}")
            except OSError as exc:
                self.status_label.setText(f"Failed to move to null: {exc}")
        else:
            self.status_label.setText("Edit cancelled.")

        if self.inspect_mode:
            self._display_inspect_image()
            self.timer.stop()

    def _set_display_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Store and render a pixmap scaled to the video label."""
        self._last_pixmap = pixmap
        self._apply_display_pixmap()

    def _apply_display_pixmap(self) -> None:
        """Apply the stored pixmap to the label with aspect fit so it stays centered on resize."""
        if self._last_pixmap is None:
            return
        target_size = self.video_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._last_pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        """Keep the displayed image centered and scaled on resize."""
        self._apply_display_pixmap()
        super().resizeEvent(event)

    def update_frame(self) -> None:
        """Grab a frame from OpenCV, convert it, and display it in the QLabel."""
        if self.inspect_mode:
            return
        if not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok:
            self.video_label.setText("Failed to read frame from camera.")
            self.timer.stop()
            return

        # Store the latest camera frame so the capture button can save it.
        self.current_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width

        q_image = QtGui.QImage(
            frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self._set_display_pixmap(pixmap)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        """Release the camera when the window closes to free the device."""
        # Pause the live feed while we handle capture + labeling.
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
        elif event.key() in (QtCore.Qt.Key_Q,):
            self.close()
        elif self.inspect_mode and event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            if self.inspect_files:
                delta = -1 if event.key() == QtCore.Qt.Key_Left else 1
                self.inspect_index = (self.inspect_index + delta) % len(self.inspect_files)
                self._display_inspect_image()
        elif self.inspect_mode and event.key() in (QtCore.Qt.Key_Delete,):
            self._delete_current_inspect()
        elif self.inspect_mode and event.key() in (QtCore.Qt.Key_E,):
            self._edit_current_inspect()
        elif self.inspect_mode and event.key() in (QtCore.Qt.Key_Escape,):
            self.exit_inspect_mode()
        else:
            super().keyPressEvent(event)

    def capture_frame(self) -> None:
        """Save the most recent frame to disk, then force labeling before continuing."""
        if self.inspect_mode:
            self.status_label.setText("Exit inspect mode before capturing.")
            return
        if self.current_frame is None:
            self.status_label.setText("No frame available yet.")
            return

        self.timer.stop()
        self.capture_button.setEnabled(False)

        # Always save into the captures folder using a timestamp so files never clash.
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

        # Force the label step so we never keep unlabeled images around.
        result = annotate_image(image_path, self.classes, self.class_colors)
        if result == "saved":
            self.status_label.setText(f"Labeled {image_path.name}.")
        elif result == "null":
            NULL_DIR.mkdir(parents=True, exist_ok=True)
            dest = NULL_DIR / image_path.name
            try:
                # "Null" means no defects; move the image away from training data.
                image_path.replace(dest)
                image_path.with_suffix(".txt").unlink(missing_ok=True)
                try:
                    rel = dest.relative_to(Path.cwd())
                except ValueError:
                    rel = dest
                self.status_label.setText(f"Marked null -> {rel}")
            except OSError as exc:
                self.status_label.setText(f"Failed to move to null: {exc}")
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


class CameraSettingsDialog(QtWidgets.QDialog):
    """Dialog to adjust camera properties like exposure, white balance, and gain."""

    def __init__(self, cap: cv2.VideoCapture, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.cap = cap
        self._prop_ids = {
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "wb": getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45),
            "gain": cv2.CAP_PROP_GAIN,
        }
        self._original_values = self._snapshot_properties()

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(-13.0, 13.0)
        self.exposure_spin.setSingleStep(0.1)
        exp_supported, _ = self._prepare_field(
            self.exposure_spin,
            self._prop_ids["exposure"],
            default=0.0,
            probe_set=False,
            restore_after_probe=False,
        )
        form.addRow("Exposure", self.exposure_spin)

        self.wb_spin = QtWidgets.QSpinBox()
        self.wb_spin.setRange(2000, 10000)
        self.wb_spin.setSingleStep(100)
        wb_supported, _ = self._prepare_field(
            self.wb_spin,
            self._prop_ids["wb"],
            default=4500,
            probe_set=True,
            restore_after_probe=True,
            zero_means_unsupported=True,
        )
        form.addRow("White Balance (K)", self.wb_spin)

        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 255.0)
        self.gain_spin.setSingleStep(1.0)
        gain_supported, _ = self._prepare_field(
            self.gain_spin,
            self._prop_ids["gain"],
            default=0.0,
            probe_set=True,
            restore_after_probe=True,
            zero_means_unsupported=True,
        )
        form.addRow("Gain", self.gain_spin)

        unsupported_note = QtWidgets.QLabel("Unavailable controls are disabled based on camera support.")
        unsupported_note.setStyleSheet("color: gray; font-size: 11px;")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(unsupported_note)
        layout.addWidget(buttons)
        self.setLayout(layout)

        # Store support flags for use when applying.
        self._supports = {
            "exposure": exp_supported,
            "wb": wb_supported,
            "gain": gain_supported,
        }

        # Apply live while dialog is open.
        if exp_supported:
            self.exposure_spin.valueChanged.connect(lambda v: self._apply_live("exposure", float(v)))
        if wb_supported:
            self.wb_spin.valueChanged.connect(lambda v: self._apply_live("wb", float(v)))
        if gain_supported:
            self.gain_spin.valueChanged.connect(lambda v: self._apply_live("gain", float(v)))

    def _prepare_field(
        self,
        widget: QtWidgets.QAbstractSpinBox,
        prop_id: int,
        default: float,
        probe_set: bool = False,
        zero_means_unsupported: bool = False,
        restore_after_probe: bool = False,
    ) -> Tuple[bool, Optional[float]]:
        """Populate a field from the camera; disable if unsupported."""
        supported, value = self._read_property(
            prop_id,
            probe_set=probe_set,
            zero_means_unsupported=zero_means_unsupported,
            restore_after_probe=restore_after_probe,
        )
        if not supported:
            widget.setEnabled(False)
            widget.setToolTip("Not supported by this camera/driver.")
            self._set_spin_value(widget, default)
        else:
            self._set_spin_value(widget, value if value is not None else default)
        return supported, value

    def _read_property(
        self,
        prop_id: int,
        probe_set: bool = False,
        zero_means_unsupported: bool = False,
        restore_after_probe: bool = False,
    ) -> Tuple[bool, Optional[float]]:
        """Check if property is supported by reading it; optionally probe a no-op set."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        value = self.cap.get(prop_id)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False, None
        # If probing is disabled, treat zero-as-unsupported heuristically.
        if not probe_set and zero_means_unsupported and value == 0:
            return False, None
        # Probe with a no-op set; some drivers report values but reject writes.
        if probe_set:
            original_value = value
            ok = self.cap.set(prop_id, value)
            if not ok:
                return False, None
            if restore_after_probe and original_value is not None:
                # Restore original value in case the driver reset it (e.g., to -6).
                self.cap.set(prop_id, original_value)
                restored = self.cap.get(prop_id)
                if restored is None or (isinstance(restored, float) and math.isnan(restored)):
                    return False, None
                if not math.isclose(restored, original_value, rel_tol=1e-3, abs_tol=1e-3):
                    # Try one more restore; if it fails, mark unsupported to avoid side effects.
                    self.cap.set(prop_id, original_value)
                    restored = self.cap.get(prop_id)
                    if not math.isclose(restored, original_value, rel_tol=1e-3, abs_tol=1e-3):
                        return False, None
        return True, value

    @staticmethod
    def _set_spin_value(widget: QtWidgets.QAbstractSpinBox, value: float) -> None:
        """Set a spin box value with correct type for int vs. double spin boxes."""
        if isinstance(widget, QtWidgets.QSpinBox):
            widget.setValue(int(round(value)))
        else:
            widget.setValue(float(value))

    def apply_settings(self) -> None:
        """Write the chosen settings back to the camera."""
        if not self.cap or not self.cap.isOpened():
            return
        if self._supports.get("exposure"):
            self.cap.set(self._prop_ids["exposure"], float(self.exposure_spin.value()))
        if self._supports.get("wb"):
            self.cap.set(self._prop_ids["wb"], float(self.wb_spin.value()))
        if self._supports.get("gain"):
            self.cap.set(self._prop_ids["gain"], float(self.gain_spin.value()))

    def _apply_live(self, prop_key: str, value: float) -> None:
        """Apply a property immediately to give live preview feedback."""
        if not self.cap or not self.cap.isOpened():
            return
        if not self._supports.get(prop_key):
            return
        prop_id = self._prop_ids.get(prop_key)
        if prop_id is None:
            return
        self.cap.set(prop_id, value)

    def _snapshot_properties(self) -> dict:
        """Capture current camera properties to restore on cancel."""
        snapshot = {}
        if not self.cap or not self.cap.isOpened():
            return snapshot
        for key, prop_id in self._prop_ids.items():
            val = self.cap.get(prop_id)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            snapshot[key] = val
        return snapshot

    def restore_original_settings(self) -> None:
        """Revert camera properties to their original values."""
        if not self.cap or not self.cap.isOpened():
            return
        for key, val in self._original_values.items():
            prop_id = self._prop_ids.get(key)
            if prop_id is None:
                continue
            self.cap.set(prop_id, val)

    def reject(self) -> None:  # type: ignore[override]
        """On cancel/close, restore original settings."""
        self.restore_original_settings()
        super().reject()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display a live camera feed in a PyQt window.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Optional frame width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Optional frame height in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app_icon = load_app_icon()
    if app_icon:
        app.setWindowIcon(app_icon)

    # Lightweight loading splash to give immediate feedback on launch.
    splash_pix = QtGui.QPixmap(460, 420)
    splash_pix.fill(QtGui.QColor("#f0f0f0"))
    painter = QtGui.QPainter(splash_pix)
    painter.setPen(QtGui.QColor("#000000"))

    # Title at top
    title_font = QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold)
    painter.setFont(title_font)
    painter.drawText(splash_pix.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, WINDOW_TITLE)

    # Large white logo in center
    logo = logo_pixmap(QtGui.QColor("#000000"))
    scaled_logo = logo.scaled(174, 174, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    logo_x = (splash_pix.width() - scaled_logo.width()) // 2
    logo_y = 100
    painter.drawPixmap(logo_x, logo_y, scaled_logo)

    # Footer text at bottom
    painter.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Normal))
    footer_text = f"Loading {WINDOW_TITLE}...\nBuild: {BUILD_DATE} | JShade.co.uk"
    text_y = logo_y + scaled_logo.height() + 20
    text_height = max(40, splash_pix.height() - text_y - 20)
    painter.drawText(
        0,
        text_y,
        splash_pix.width(),
        text_height,
        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        footer_text,
    )
    painter.end()
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    window = CameraWindow(args.camera, args.width, args.height)
    if app_icon:
        window.setWindowIcon(app_icon)
    window.show()
    splash.finish(window)

    # Start the Qt event loop; this keeps the window responsive until closed.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
