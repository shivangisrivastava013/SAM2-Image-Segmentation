import argparse
import os

import numpy as np

from sam2_pipeline.evaluation import compute_boundary_f1, compute_dice, compute_iou
from sam2_pipeline.pipeline import YOLOGuidedSAM2Pipeline
from sam2_pipeline.sam2_engine import SAM2SegmentationEngine
from sam2_pipeline.visualization import render_segmentation_overlay
from sam2_pipeline.yolo_detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description="SAM2 Image Segmentation & YOLO-Guided Pipeline Demonstration.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_small.pt", help="Checkpoint path.")
    parser.add_argument("--mock-mode", action="store_true", help="Run in mock mode without downloading weights.")

    args = parser.parse_args()

    # Check if checkpoint exists; if not, force mock_mode for demonstration
    mock_mode = args.mock_mode or not os.path.exists(args.checkpoint)

    print("[+] Initializing SAM2 Image Segmentation Pipeline...")
    if mock_mode:
        print("[!] NOTICE: SAM2 checkpoint weights not detected or mock_mode enabled. Operating in explicit mock mode.")
    else:
        print(f"[+] Loading SAM2 model weights from '{args.checkpoint}'...")

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./examples/input", exist_ok=True)

    # Generate synthetic input image (gradient ellipse)
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    cy, cx, ry, rx = 128, 128, 55, 75
    mask_gt = (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0).astype(np.float32)

    image_np = np.zeros((h, w, 3), dtype=np.uint8)
    image_np[mask_gt > 0] = [6, 182, 212]  # Cyan object

    # Initialize Engine & Pipeline
    sam2 = SAM2SegmentationEngine(checkpoint_path=args.checkpoint, mock_mode=mock_mode)
    detector = YOLODetector(mock_mode=mock_mode)
    pipeline = YOLOGuidedSAM2Pipeline(detector=detector, sam2_engine=sam2, mock_mode=mock_mode)

    # Execute YOLO-Guided SAM2 Segmentation
    dets, segs = pipeline.segment_instances(image_np)

    iou = compute_iou(segs[0].mask, mask_gt) if segs else 0.0
    dice = compute_dice(segs[0].mask, mask_gt) if segs else 0.0
    bf1 = compute_boundary_f1(segs[0].mask, mask_gt) if segs else 0.0

    print("\n[*] SEGMENTATION BENCHMARK METRICS:")
    print(f"   - Engine Source:                    {segs[0].source if segs else 'N/A'}")
    print(f"   - Mask IoU (Intersection over Union): {iou * 100:.2f}%")
    print(f"   - Dice Similarity Coefficient:       {dice * 100:.2f}%")
    print(f"   - Boundary F1 Score:                {bf1 * 100:.2f}%")
    if segs:
        print(f"   - Inference Latency:                {segs[0].inference_ms:.2f} ms")

    output_path = "./results/sam2_demo_result.png"
    render_segmentation_overlay(
        image_np,
        segs,
        dets,
        output_path=output_path,
        title=f"YOLO-Guided SAM2 (IoU: {iou*100:.1f}%) [{segs[0].source}]",
    )

    print(f"\n[+] Visual overlay output saved to: {output_path}")


if __name__ == "__main__":
    main()
