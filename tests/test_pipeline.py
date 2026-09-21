import numpy as np
import pytest

from sam2_pipeline.pipeline import YOLOGuidedSAM2Pipeline


def test_pipeline_execution():
    pipeline = YOLOGuidedSAM2Pipeline(mock_mode=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    dets, segs = pipeline.segment_instances(img)
    assert len(dets) == len(segs)
    assert len(segs) > 0
    assert segs[0].source == "synthetic_mock"
    assert segs[0].prompt_type == "yolo_box"


def test_invalid_image_shape_raises_error():
    pipeline = YOLOGuidedSAM2Pipeline(mock_mode=True)
    img_invalid = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="image_np must be RGB array"):
        pipeline.segment_instances(img_invalid)
