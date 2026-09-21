from dataclasses import dataclass, field
from typing import List


@dataclass
class SAM2Config:
    """
    Configuration options for SAM2 Image Segmentation Pipeline.
    """

    model_cfg: str = "sam2_hiera_s.yaml"
    checkpoint_path: str = "checkpoints/sam2_hiera_small.pt"
    device: str = "cuda"  # 'cuda' or 'cpu'
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.45
    safety_classes: List[str] = field(
        default_factory=lambda: ["person", "car", "bicycle", "bus", "truck", "dog", "chair", "bottle"]
    )
    mock_mode: bool = False

    def validate(self) -> None:
        """
        Validates configuration options.
        """
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        if self.iou_threshold < 0.0 or self.iou_threshold > 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")
