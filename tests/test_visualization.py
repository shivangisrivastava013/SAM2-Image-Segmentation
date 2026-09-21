import os

import numpy as np

from sam2_pipeline.schemas import SegmentationResult
from sam2_pipeline.visualization import render_segmentation_overlay


def test_render_segmentation_overlay(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0

    seg = SegmentationResult(
        mask=mask,
        score=0.95,
        prompt_type="box",
        inference_ms=10.0,
        model_name="sam2_hiera_s.yaml",
        source="synthetic_mock",
        bbox=[25, 25, 75, 75],
        class_name="person",
    )

    out_file = str(tmp_path / "overlay.png")
    out_img = render_segmentation_overlay(img, [seg], output_path=out_file)

    assert os.path.exists(out_file)
    assert out_img.ndim == 3
