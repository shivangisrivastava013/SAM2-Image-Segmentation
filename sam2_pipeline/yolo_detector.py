import logging
import time
from typing import List, Optional

import numpy as np

from .schemas import DetectionResult

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Object detector wrapper around Ultralytics YOLOv8.
    Strictly raises RuntimeError if model fails to load unless mock_mode=True is set.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.50,
        mock_mode: bool = False,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.mock_mode = mock_mode
        self.model = None

        if not self.mock_mode:
            self._init_model()

    def _init_model(self) -> None:
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model '{self.model_name}'...")
            self.model = YOLO(self.model_name)
            logger.info("YOLO detector initialized.")
        except Exception as e:
            raise RuntimeError(
                f"YOLO model '{self.model_name}' could not be loaded: {e}. "
                "Use mock_mode=True explicitly for simulated object detections."
            ) from e

    def detect(self, image_np: np.ndarray, safety_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        Runs object detection on image_np [H, W, 3] uint8.
        Filters by confidence_threshold and optional safety_classes.
        """
        t0 = time.perf_counter()
        if self.mock_mode:
            return self._generate_mock_detections(image_np, safety_classes)

        if self.model is None:
            raise RuntimeError("YOLO model is uninitialized.")

        results = self.model.predict(image_np, conf=self.confidence_threshold, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[DetectionResult] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return detections

        names = res.names
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            if safety_classes and cls_name not in safety_classes:
                continue

            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=xyxy,
                    source=self.model_name,
                    inference_ms=latency_ms / max(len(boxes), 1),
                )
            )

        return detections

    def _generate_mock_detections(
        self, image_np: np.ndarray, safety_classes: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        h, w = image_np.shape[:2]
        all_mocks = [
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.91,
                bbox=[float(w * 0.2), float(h * 0.2), float(w * 0.8), float(h * 0.8)],
                source="synthetic_mock",
                inference_ms=12.5,
            )
        ]
        if safety_classes:
            return [d for d in all_mocks if d.class_name in safety_classes]
        return all_mocks
