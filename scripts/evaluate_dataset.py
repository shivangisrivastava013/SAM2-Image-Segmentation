import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2_pipeline.evaluation import run_benchmark_eval


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 automated benchmark evaluation suite.")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/sam2_hiera_small.pt", help="Path to SAM2 checkpoint."
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for saving benchmark results.")
    parser.add_argument("--mock-mode", action="store_true", help="Execute evaluation using mock mode.")

    args = parser.parse_args()

    results = run_benchmark_eval(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        mock_mode=args.mock_mode,
    )

    title = "MOCK-MODE PIPELINE VALIDATION" if args.mock_mode else "REAL SAM2 CHECKPOINT BENCHMARK"
    print("\n========================================================")
    print(f"   SAM2 IMAGE SEGMENTATION - {title}")
    print("========================================================")
    print(f"Execution Source: {results['execution_source']}\n")

    summary = results["benchmark_summary"]
    print("| Pipeline | Mean IoU | Mean Dice | Boundary F1 | Latency (ms) | FPS |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, metrics in summary.items():
        print(
            f"| **{name}** | **{metrics['mean_iou']:.4f}** | **{metrics['mean_dice']:.4f}** | "
            f"**{metrics['boundary_f1']:.4f}** | {metrics['latency_ms']:.1f} | {metrics['fps']:.1f} |"
        )
    print("========================================================\n")


if __name__ == "__main__":
    main()
