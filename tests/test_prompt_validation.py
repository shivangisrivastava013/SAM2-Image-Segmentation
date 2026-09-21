import numpy as np
import pytest

from sam2_pipeline.sam2_engine import SAM2SegmentationEngine


def test_invalid_point_coords_shape():
    engine = SAM2SegmentationEngine(mock_mode=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="point_coords must be of shape"):
        engine.predict_from_points(img, np.array([50, 50]), np.array([1]))


def test_invalid_box_coordinates():
    engine = SAM2SegmentationEngine(mock_mode=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid bounding box coordinates"):
        engine.predict_from_box(img, [80, 80, 20, 20])
