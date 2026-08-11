import os
import numpy as np
import matplotlib.pyplot as plt
from sam2_pipeline.model import SAM2SegmentationEngine
from sam2_pipeline.eval import compute_iou, compute_dice


def main():
    print("[+] Initializing SAM2 Image Segmentation Pipeline...")
    os.makedirs("./results", exist_ok=True)

    # Generate synthetic input image (gradient circle)
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    mask_gt = ((x - 128)**2 + (y - 128)**2 <= 60**2).astype(np.float32)

    image_np = np.zeros((h, w, 3), dtype=np.uint8)
    image_np[mask_gt > 0] = [0, 242, 254]

    engine = SAM2SegmentationEngine()
    point_coords = np.array([[128, 128]])
    point_labels = np.array([1])

    pred_mask = engine.predict_mask_from_points(image_np, point_coords, point_labels)

    iou = compute_iou(pred_mask, mask_gt)
    dice = compute_dice(pred_mask, mask_gt)

    print("\n[*] SEGMENTATION METRICS:")
    print(f"   - Mask IoU (Intersection over Union): {iou * 100:.2f}%")
    print(f"   - Dice Similarity Coefficient:       {dice * 100:.2f}%")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(mask_gt, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred_mask, cmap="cyan" if hasattr(plt.cm, "cyan") else "viridis")
    axs[2].set_title(f"SAM2 Predicted Mask (IoU: {iou*100:.1f}%)")
    for ax in axs: ax.axis("off")

    plt.tight_layout()
    output_path = "./results/sam2_demo_result.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n[+] Demo output saved to: {output_path}")


if __name__ == '__main__':
    main()
