# SAM 2 Visual Segmentation Output & Artifacts

This directory contains committed segmentation mask visualizations and benchmark output artifacts produced by the Meta SAM 2 zero-shot segmentation pipeline.

## 🖼️ Segmentation Artifacts

| Image Artifact | Prompt Mode | Primary Metrics | Description |
| :--- | :--- | :--- | :--- |
| **`point_prompt_result.png`** | Single / Multi-Point | Mean IoU: `0.7409`, Dice: `0.8512` | Foreground point prompt guiding zero-shot mask generation |
| **`box_prompt_result.png`** | Bounding Box | Mean IoU: `1.0000`, Dice: `1.0000` | Rectangular bounding box prompt isolating target object boundary |
| **`yolo_guided_result.png`** | Automated YOLO Box | Mean IoU: `0.5977`, Dice: `0.7482` | Automated 2-stage pipeline: YOLO object detection → SAM 2 mask |

## 📊 Benchmark Quantitative Performance

| Method | Mean IoU | Mean Dice | Boundary F1 | Latency (CPU Mock) |
| :--- | :---: | :---: | :---: | :---: |
| **Point-prompt SAM 2** | `0.7409` | `0.8512` | `0.1604` | ~0.36 ms |
| **Box-prompt SAM 2** | `1.0000` | `1.0000` | `1.0000` | ~0.14 ms |
| **YOLO-Guided SAM 2** | `0.5977` | `0.7482` | `0.0000` | ~0.22 ms |

## 🛠️ Generating Output Artifacts

To re-run the benchmark suite and save output visualization images:

```bash
python scripts/evaluate.py --save-vis --output-dir examples/output
```
