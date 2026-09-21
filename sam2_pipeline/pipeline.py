import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .sam2_engine import SAM2SegmentationEngine
from .schemas import DetectionResult, SegmentationResult
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class YOLOGuidedSAM2Pipeline:
    """
    End-to-End Object Instance Segmentation Pipeline:
    Uses YOLOv8 to detect candidate object bounding boxes, then prompts SAM2
    to extract precise instance segmentation masks.
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        sam2_engine: Optional[SAM2SegmentationEngine] = None,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self.detector = detector or YOLODetector(mock_mode=mock_mode)
        self.sam2_engine = sam2_engine or SAM2SegmentationEngine(mock_mode=mock_mode)

    def segment_instances(
        self,
        image_np: np.ndarray,
        safety_classes: Optional[List[str]] = None,
        max_objects: int = 10,
    ) -> Tuple[List[DetectionResult], List[SegmentationResult]]:
        """
        Detects objects in image_np and generates corresponding SAM2 instance masks.
        """
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image_np must be RGB array of shape [H, W, 3], got {image_np.shape}")

        t0 = time.perf_counter()
        detections = self.detector.detect(image_np, safety_classes=safety_classes)
        if len(detections) > max_objects:
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:max_objects]

        segmentations: List[SegmentationResult] = []
        for det in detections:
            seg_res = self.sam2_engine.predict_from_box(image_np, det.bbox)
            seg_res.class_name = det.class_name
            seg_res.prompt_type = "yolo_box"
            segmentations.append(seg_res)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Pipeline processed {len(detections)} instances in {elapsed_ms:.1f} ms.")

        return detections, segmentations
