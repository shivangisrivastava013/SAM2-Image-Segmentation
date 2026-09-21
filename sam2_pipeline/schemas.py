from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """
    Structured object detection result from YOLO.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    source: str = "yolov8n.pt"
    inference_ms: float = 0.0


@dataclass
class SegmentationResult:
    """
    Structured segmentation result from SAM2.
    """

    mask: np.ndarray  # Binary mask [H, W] float32/bool
    score: float
    prompt_type: str  # 'point', 'box', 'points_and_box', 'yolo_box', 'auto'
    inference_ms: float
    model_name: str
    source: str  # 'sam2_checkpoint' or 'synthetic_mock'
    bbox: Optional[List[float]] = None
    class_name: Optional[str] = None
