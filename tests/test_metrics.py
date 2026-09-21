import numpy as np
import pytest

from sam2_pipeline.evaluation import compute_boundary_f1, compute_dice, compute_iou


def test_iou_perfect_match():
    mask = np.ones((100, 100), dtype=np.float32)
    assert compute_iou(mask, mask) == pytest.approx(1.0)


def test_iou_disjoint_masks():
    mask_a = np.zeros((100, 100), dtype=np.float32)
    mask_a[:50, :50] = 1.0
    mask_b = np.zeros((100, 100), dtype=np.float32)
    mask_b[50:, 50:] = 1.0

    assert compute_iou(mask_a, mask_b) == pytest.approx(0.0)


def test_dice_perfect_match():
    mask = np.ones((100, 100), dtype=np.float32)
    assert compute_dice(mask, mask) == pytest.approx(1.0)


def test_boundary_f1_perfect_match():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    assert compute_boundary_f1(mask, mask) == pytest.approx(1.0)
