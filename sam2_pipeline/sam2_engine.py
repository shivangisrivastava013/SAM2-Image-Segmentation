import logging
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .schemas import SegmentationResult

logger = logging.getLogger(__name__)


class SAM2SegmentationEngine:
    """
    Zero-Shot Visual Segmentation Engine powered by Meta's Segment Anything Model 2 (SAM2).
    Strictly enforces checkpoint validation unless mock_mode=True is explicitly enabled.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_s.yaml",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.mock_mode = mock_mode
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.predictor = None

        if not self.mock_mode:
            self._init_model()

    def _init_model(self) -> None:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise RuntimeError(
                f"SAM2 checkpoint unavailable at '{self.checkpoint_path}'. "
                "Download weights via 'python scripts/download_checkpoint.py' "
                "or pass mock_mode=True explicitly for testing."
            )

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading SAM2 model '{self.model_cfg}' from '{self.checkpoint_path}' on {self.device}...")
            model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(model)
            logger.info("SAM2 predictor successfully initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model from '{self.checkpoint_path}': {e}") from e

    def predict_from_points(
        self,
        image_np: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> SegmentationResult:
        """
        Generates segmentation mask from point prompts [[x, y]].
        """
        if point_coords.ndim != 2 or point_coords.shape[1] != 2:
            raise ValueError(f"point_coords must be of shape [N, 2], got {point_coords.shape}")
        if point_labels.ndim != 1 or point_labels.shape[0] != point_coords.shape[0]:
            raise ValueError(f"point_labels length must match point_coords count {point_coords.shape[0]}")

        t0 = time.perf_counter()
        if self.mock_mode:
            mask, score = self._generate_mock_point_mask(image_np, point_coords)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                mask=mask,
                score=score,
                prompt_type="point",
                inference_ms=latency_ms,
                model_name=self.model_cfg,
                source="synthetic_mock",
            )

        if self.predictor is None:
            raise RuntimeError("SAM2 predictor is uninitialized.")

        self.predictor.set_image(image_np)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SegmentationResult(
            mask=masks[0].astype(np.float32),
            score=float(scores[0]),
            prompt_type="point",
            inference_ms=latency_ms,
            model_name=self.model_cfg,
            source="sam2_checkpoint",
        )

    def predict_from_box(
        self,
        image_np: np.ndarray,
        box: Union[List[float], np.ndarray],
    ) -> SegmentationResult:
        """
        Generates segmentation mask from bounding box prompt [xmin, ymin, xmax, ymax].
        """
        box_arr = np.array(box, dtype=np.float32).flatten()
        if box_arr.shape[0] != 4:
            raise ValueError(f"Bounding box prompt must be [xmin, ymin, xmax, ymax], got {box}")
        if box_arr[0] >= box_arr[2] or box_arr[1] >= box_arr[3]:
            raise ValueError(f"Invalid bounding box coordinates: {box_arr}")

        t0 = time.perf_counter()
        if self.mock_mode:
            mask, score = self._generate_mock_box_mask(image_np, box_arr)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                mask=mask,
                score=score,
                prompt_type="box",
                inference_ms=latency_ms,
                model_name=self.model_cfg,
                source="synthetic_mock",
                bbox=box_arr.tolist(),
            )

        if self.predictor is None:
            raise RuntimeError("SAM2 predictor is uninitialized.")

        self.predictor.set_image(image_np)
        masks, scores, _ = self.predictor.predict(
            box=box_arr[None, :],
            multimask_output=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SegmentationResult(
            mask=masks[0].astype(np.float32),
            score=float(scores[0]),
            prompt_type="box",
            inference_ms=latency_ms,
            model_name=self.model_cfg,
            source="sam2_checkpoint",
            bbox=box_arr.tolist(),
        )

    def predict_from_points_and_box(
        self,
        image_np: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box: Union[List[float], np.ndarray],
    ) -> SegmentationResult:
        """
        Generates segmentation mask from combined point and box prompts.
        """
        box_arr = np.array(box, dtype=np.float32).flatten()
        t0 = time.perf_counter()
        if self.mock_mode:
            mask, score = self._generate_mock_box_mask(image_np, box_arr)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                mask=mask,
                score=score,
                prompt_type="points_and_box",
                inference_ms=latency_ms,
                model_name=self.model_cfg,
                source="synthetic_mock",
                bbox=box_arr.tolist(),
            )

        if self.predictor is None:
            raise RuntimeError("SAM2 predictor is uninitialized.")

        self.predictor.set_image(image_np)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr[None, :],
            multimask_output=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SegmentationResult(
            mask=masks[0].astype(np.float32),
            score=float(scores[0]),
            prompt_type="points_and_box",
            inference_ms=latency_ms,
            model_name=self.model_cfg,
            source="sam2_checkpoint",
            bbox=box_arr.tolist(),
        )

    def _generate_mock_point_mask(self, image_np: np.ndarray, point_coords: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
        r = min(h, w) * 0.2
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        mask[dist <= r] = 1.0
        return mask, 0.92

    def _generate_mock_box_mask(self, image_np: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # Create an elliptical mask inside the bounding box
        if x2 > x1 and y2 > y1:
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            ry, rx = (y2 - y1) / 2.0, (x2 - x1) / 2.0
            y_idx, x_idx = np.ogrid[:h, :w]
            ellipse = ((x_idx - cx) / max(rx, 1e-5)) ** 2 + ((y_idx - cy) / max(ry, 1e-5)) ** 2
            mask[ellipse <= 1.0] = 1.0
        return mask, 0.95
