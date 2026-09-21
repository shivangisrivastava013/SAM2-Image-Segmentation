import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2_pipeline.pipeline import YOLOGuidedSAM2Pipeline
from sam2_pipeline.sam2_engine import SAM2SegmentationEngine
from sam2_pipeline.yolo_detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 / YOLO-Guided SAM2 segmentation on a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, default="results/segmented_video.mp4", help="Path to save video.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_small.pt", help="SAM2 checkpoint.")
    parser.add_argument("--mock-mode", action="store_true", help="Run with explicit mock engine for testing.")

    args = parser.parse_args()

    print(f"[+] Video processing CLI initialized for '{args.input}' -> '{args.output}'.")
    sam2 = SAM2SegmentationEngine(checkpoint_path=args.checkpoint, mock_mode=args.mock_mode)
    detector = YOLODetector(mock_mode=args.mock_mode)
    pipeline = YOLOGuidedSAM2Pipeline(detector=detector, sam2_engine=sam2, mock_mode=args.mock_mode)

    print(
        f"[+] Pipeline '{type(pipeline).__name__}' configured. Engine source: {'synthetic_mock' if args.mock_mode else 'sam2_checkpoint'}."
    )


if __name__ == "__main__":
    main()
