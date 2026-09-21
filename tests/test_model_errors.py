import pytest

from sam2_pipeline.sam2_engine import SAM2SegmentationEngine
from sam2_pipeline.yolo_detector import YOLODetector


def test_missing_sam2_checkpoint_raises_error():
    with pytest.raises(RuntimeError, match="SAM2 checkpoint unavailable"):
        SAM2SegmentationEngine(checkpoint_path="non_existent_path.pt", mock_mode=False)


def test_mock_mode_allows_uninitialized_weights():
    engine = SAM2SegmentationEngine(checkpoint_path=None, mock_mode=True)
    assert engine.mock_mode is True


def test_missing_yolo_model_raises_error():
    with pytest.raises(RuntimeError, match="could not be loaded"):
        YOLODetector(model_name="non_existent_yolo.pt", mock_mode=False)
