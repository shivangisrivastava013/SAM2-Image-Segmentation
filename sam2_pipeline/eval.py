import numpy as np


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) metric between predicted and ground truth masks.
    """
    intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
    union = np.logical_or(pred_mask > 0.5, gt_mask > 0.5).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Dice similarity coefficient between predicted and ground truth masks.
    """
    intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
    total = (pred_mask > 0.5).sum() + (gt_mask > 0.5).sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)
