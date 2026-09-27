"""YOLO object detection helpers."""

from typing import Any

from ultralytics import YOLO


class ObjectDetector:
    """Load a lightweight YOLO model and format its detections."""

    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        try:
            self.model = YOLO(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Could not load {model_name}. Check your internet connection "
                "the first time you run the app so Ultralytics can download the model."
            ) from error

    @staticmethod
    def format_results(results: Any) -> list[dict[str, Any]]:
        """Convert an Ultralytics result into simple detection dictionaries."""
        detections: list[dict[str, Any]] = []
        if results is None or results.boxes is None:
            return detections

        boxes = results.boxes
        names = results.names
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[index].item())

            detections.append(
                {
                    "box": [float(value) for value in box.xyxy[0].tolist()],
                    "class_id": class_id,
                    "class_name": str(names[class_id]),
                    "confidence": float(box.conf[0].item()),
                    "track_id": track_id,
                }
            )
        return detections

    def detect(self, frame: Any, confidence: float = 0.25) -> list[dict[str, Any]]:
        """Run a one-frame detection without tracking."""
        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
        )[0]
        return self.format_results(results)