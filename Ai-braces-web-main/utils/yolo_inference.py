import cv2
import numpy as np
from ultralytics import YOLO

_model = None


def load_yolo(model_path: str):
    """Load YOLO model once at startup. Call this from app.py."""
    global _model
    _model = YOLO(model_path)
    print(f"[YOLO] Model loaded from {model_path}")


def run_yolo(image_path: str):
    """
    Run YOLOv8 inference to detect brackets in a dental image.

    Args:
        image_path: Path to input image.

    Returns:
        dict with keys:
            boxes        – list of dicts: {x1, y1, x2, y2, conf, cx, cy}
            plotted_img  – BGR image with bounding boxes drawn
    """
    if _model is None:
        raise RuntimeError("YOLO model not loaded. Call load_yolo() first.")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_resized = cv2.resize(img_bgr, (512, 512))

    results = _model(img_resized, verbose=False)[0]

    plotted_img = img_resized.copy()
    boxes = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "conf": conf, "cx": cx, "cy": cy})

            # Draw bounding box
            cv2.rectangle(plotted_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"bracket {conf:.2f}"
            cv2.putText(plotted_img, label, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            # Center dot
            cv2.circle(plotted_img, (cx, cy), 4, (0, 165, 255), -1)

    return {
        "boxes":       boxes,
        "plotted_img": plotted_img,
    }
