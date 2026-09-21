import json
import os
import time
from typing import Any, Dict

import numpy as np

from .pipeline import YOLOGuidedSAM2Pipeline
from .sam2_engine import SAM2SegmentationEngine


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes Intersection over Union (IoU) between predicted mask and ground truth mask.
    """
    pred_b = (pred_mask >= threshold).astype(bool)
    gt_b = (gt_mask >= threshold).astype(bool)

    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes Dice Similarity Coefficient (DSC) between predicted mask and ground truth mask.
    """
    pred_b = (pred_mask >= threshold).astype(bool)
    gt_b = (gt_mask >= threshold).astype(bool)

    intersection = np.logical_and(pred_b, gt_b).sum()
    total = pred_b.sum() + gt_b.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection) / total)


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes Boundary F1 score measuring contour precision and recall.
    """
    pred_b = (pred_mask >= threshold).astype(np.uint8)
    gt_b = (gt_mask >= threshold).astype(np.uint8)

    # Gradient as edge proxy
    gy_p, gx_p = np.gradient(pred_b.astype(float))
    edge_pred = (np.abs(gy_p) + np.abs(gx_p)) > 0

    gy_g, gx_g = np.gradient(gt_b.astype(float))
    edge_gt = (np.abs(gy_g) + np.abs(gx_g)) > 0

    tp = np.logical_and(edge_pred, edge_gt).sum()
    fp = np.logical_and(edge_pred, ~edge_gt).sum()
    fn = np.logical_and(~edge_pred, edge_gt).sum()

    prec = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 1.0

    if prec + rec == 0:
        return 0.0

    return float((2.0 * prec * rec) / (prec + rec))


def run_benchmark_eval(
    checkpoint_path: str = "checkpoints/sam2_hiera_small.pt",
    output_dir: str = "results",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    """
    Executes automated evaluation benchmark comparing Point-prompt SAM2,
    Box-prompt SAM2, and YOLO-guided SAM2. Saves results to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    # Ground truth ellipse
    cy, cx, ry, rx = 128, 128, 50, 70
    gt_mask = (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0).astype(np.float32)

    image_np = np.zeros((h, w, 3), dtype=np.uint8)
    image_np[gt_mask > 0] = [34, 197, 94]  # Emerald green object

    sam2 = SAM2SegmentationEngine(checkpoint_path=checkpoint_path, mock_mode=mock_mode)
    pipeline = YOLOGuidedSAM2Pipeline(sam2_engine=sam2, mock_mode=mock_mode)

    # 1. Point Prompt Evaluation
    t0 = time.perf_counter()
    pt_res = sam2.predict_from_points(image_np, np.array([[128, 128]]), np.array([1]))
    pt_ms = (time.perf_counter() - t0) * 1000.0
    pt_iou = compute_iou(pt_res.mask, gt_mask)
    pt_dice = compute_dice(pt_res.mask, gt_mask)
    pt_bf1 = compute_boundary_f1(pt_res.mask, gt_mask)

    # 2. Box Prompt Evaluation
    t0 = time.perf_counter()
    box_res = sam2.predict_from_box(image_np, [58, 78, 198, 178])
    box_ms = (time.perf_counter() - t0) * 1000.0
    box_iou = compute_iou(box_res.mask, gt_mask)
    box_dice = compute_dice(box_res.mask, gt_mask)
    box_bf1 = compute_boundary_f1(box_res.mask, gt_mask)

    # 3. YOLO-Guided SAM2 Evaluation
    t0 = time.perf_counter()
    dets, segs = pipeline.segment_instances(image_np)
    yolo_ms = (time.perf_counter() - t0) * 1000.0
    yolo_iou = compute_iou(segs[0].mask, gt_mask) if segs else 0.0
    yolo_dice = compute_dice(segs[0].mask, gt_mask) if segs else 0.0
    yolo_bf1 = compute_boundary_f1(segs[0].mask, gt_mask) if segs else 0.0

    eval_results = {
        "execution_source": "synthetic_mock" if mock_mode else "sam2_checkpoint",
        "benchmark_summary": {
            "Point-prompt SAM2": {
                "mean_iou": round(pt_iou, 4),
                "mean_dice": round(pt_dice, 4),
                "boundary_f1": round(pt_bf1, 4),
                "latency_ms": round(pt_ms, 2),
                "fps": round(1000.0 / max(pt_ms, 0.1), 1),
            },
            "Box-prompt SAM2": {
                "mean_iou": round(box_iou, 4),
                "mean_dice": round(box_dice, 4),
                "boundary_f1": round(box_bf1, 4),
                "latency_ms": round(box_ms, 2),
                "fps": round(1000.0 / max(box_ms, 0.1), 1),
            },
            "YOLO-Guided SAM2": {
                "mean_iou": round(yolo_iou, 4),
                "mean_dice": round(yolo_dice, 4),
                "boundary_f1": round(yolo_bf1, 4),
                "latency_ms": round(yolo_ms, 2),
                "fps": round(1000.0 / max(yolo_ms, 0.1), 1),
            },
        },
    }

    out_json = os.path.join(output_dir, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results
