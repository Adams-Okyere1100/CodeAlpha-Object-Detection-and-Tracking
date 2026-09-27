"""Frame annotation and object-counting helpers."""

from collections import Counter
from typing import Any

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw boxes, class names, confidence scores, and tracking IDs."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        color = (45, 190, 90)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        track_id = detection["track_id"]
        if track_id is not None:
            label += f" ID:{track_id}"

        text_y = max(y1 - 8, 18)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def count_objects(detections: list[dict[str, Any]]) -> Counter[str]:
    """Count detected objects by class in a single frame."""
    return Counter(detection["class_name"] for detection in detections)


def process_frame(
    frame: np.ndarray,
    tracker: Any,
    confidence: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Track and annotate one frame, returning the image and detections."""
    detections = tracker.update(frame, confidence)
    return draw_detections(frame, detections), detections