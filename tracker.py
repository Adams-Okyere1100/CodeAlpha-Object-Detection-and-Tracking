"""Persistent ByteTrack tracking through the Ultralytics API."""

from typing import Any

from detector import ObjectDetector


class ObjectTracker:
    """Track detections across consecutive video frames using ByteTrack."""

    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector

    def update(self, frame: Any, confidence: float = 0.25) -> list[dict[str, Any]]:
        """Track one frame; persist=True keeps IDs consistent across frames."""
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False,
        )[0]
        return self.detector.format_results(results)