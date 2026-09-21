import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2_pipeline.pipeline import YOLOGuidedSAM2Pipeline
from sam2_pipeline.sam2_engine import SAM2SegmentationEngine
from sam2_pipeline.visualization import render_segmentation_overlay
from sam2_pipeline.yolo_detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 / YOLO-Guided SAM2 segmentation on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--output", type=str, default="results/segmented_image.png", help="Path to save output.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_small.pt", help="SAM2 checkpoint.")
    parser.add_argument("--config", type=str, default="sam2_hiera_s.yaml", help="SAM2 model configuration.")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO-guided automatic box prompting.")
    parser.add_argument("--prompt-type", type=str, default="box", choices=["point", "box", "yolo"], help="Prompt type.")
    parser.add_argument("--box", type=float, nargs=4, help="Bounding box prompt [xmin, ymin, xmax, ymax].")
    parser.add_argument("--points", type=float, nargs="+", help="Point coordinates x1 y1 x2 y2...")
    parser.add_argument("--mock-mode", action="store_true", help="Run with explicit mock engine for testing.")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] Input image '{args.image}' not found.")
        return

    img_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(img_pil)

    if args.use_yolo or args.prompt_type == "yolo":
        print("[+] Running YOLO-Guided SAM2 Instance Segmentation...")
        sam2 = SAM2SegmentationEngine(model_cfg=args.config, checkpoint_path=args.checkpoint, mock_mode=args.mock_mode)
        detector = YOLODetector(mock_mode=args.mock_mode)
        pipeline = YOLOGuidedSAM2Pipeline(detector=detector, sam2_engine=sam2, mock_mode=args.mock_mode)

        dets, segs = pipeline.segment_instances(image_np)
        render_segmentation_overlay(image_np, segs, dets, output_path=args.output)
        print(f"[+] Output saved to '{args.output}'. Processed {len(segs)} instance masks.")
        return

    sam2 = SAM2SegmentationEngine(model_cfg=args.config, checkpoint_path=args.checkpoint, mock_mode=args.mock_mode)

    if args.box:
        seg = sam2.predict_from_box(image_np, args.box)
        render_segmentation_overlay(image_np, [seg], output_path=args.output)
        print(f"[+] Bounding Box prompt segmentation saved to '{args.output}'. Score: {seg.score:.4f}")
    elif args.points:
        pts = np.array(args.points).reshape(-1, 2)
        lbls = np.ones(len(pts), dtype=np.int32)
        seg = sam2.predict_from_points(image_np, pts, lbls)
        render_segmentation_overlay(image_np, [seg], output_path=args.output)
        print(f"[+] Point prompt segmentation saved to '{args.output}'. Score: {seg.score:.4f}")
    else:
        # Default box prompt covering central object region
        h, w = image_np.shape[:2]
        default_box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]
        seg = sam2.predict_from_box(image_np, default_box)
        render_segmentation_overlay(image_np, [seg], output_path=args.output)
        print(f"[+] Default box prompt segmentation saved to '{args.output}'. Score: {seg.score:.4f}")


if __name__ == "__main__":
    main()
