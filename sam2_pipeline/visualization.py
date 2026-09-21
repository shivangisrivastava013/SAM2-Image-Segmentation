import os
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .schemas import DetectionResult, SegmentationResult


def render_segmentation_overlay(
    image_np: np.ndarray,
    segmentations: List[SegmentationResult],
    detections: Optional[List[DetectionResult]] = None,
    output_path: Optional[str] = None,
    title: str = "YOLO-Guided SAM2 Instance Segmentation",
) -> np.ndarray:
    """
    Renders high-quality visual overlay showing input image, bounding boxes,
    colored SAM2 instance masks, confidence badges, and segmentation stats.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)

    colors = [
        [0.0, 0.95, 1.0, 0.45],  # Cyan
        [0.1, 0.85, 0.2, 0.45],  # Green
        [1.0, 0.2, 0.4, 0.45],  # Magenta
        [1.0, 0.8, 0.0, 0.45],  # Gold
        [0.6, 0.2, 1.0, 0.45],  # Purple
    ]

    for idx, seg in enumerate(segmentations):
        color = colors[idx % len(colors)]
        mask = seg.mask >= 0.5

        # Render colored mask overlay
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
        colored_mask[mask] = color
        ax.imshow(colored_mask)

        # Draw bounding box if present
        bbox = seg.bbox
        if bbox:
            x1, y1, x2, y2 = bbox
            w_box, h_box = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                w_box,
                h_box,
                linewidth=2,
                edgecolor=color[:3],
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

            label_text = f"{seg.class_name or 'Object'} ({seg.score:.2f}) [{seg.source}]"
            ax.text(
                x1,
                max(y1 - 5, 12),
                label_text,
                color="white",
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.8),
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    out_img = np.asarray(rgba_buf)[:, :, :3]
    plt.close(fig)

    return out_img
