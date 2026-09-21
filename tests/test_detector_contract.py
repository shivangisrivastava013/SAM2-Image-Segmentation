import numpy as np

from sam2_pipeline.yolo_detector import YOLODetector


def test_detector_mock_mode_schema():
    detector = YOLODetector(mock_mode=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    dets = detector.detect(img)
    assert len(dets) > 0
    assert dets[0].class_name == "person"
    assert dets[0].source == "synthetic_mock"
    assert len(dets[0].bbox) == 4
