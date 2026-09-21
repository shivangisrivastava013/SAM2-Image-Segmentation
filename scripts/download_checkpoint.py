import argparse
import os
import urllib.request

CHECKPOINT_URLS = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


def download_checkpoint(model_name: str = "sam2_hiera_small", output_dir: str = "checkpoints") -> str:
    """
    Downloads official Meta Segment Anything 2 (SAM2) model weights.
    """
    if model_name not in CHECKPOINT_URLS:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(CHECKPOINT_URLS.keys())}")

    url = CHECKPOINT_URLS[model_name]
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, f"{model_name}.pt")

    if os.path.exists(target_path):
        print(f"[*] Checkpoint already exists at '{target_path}'. Skipping download.")
        return target_path

    print(f"[+] Downloading SAM2 weights '{model_name}' from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[+] Successfully saved SAM2 weights to '{target_path}'.")
    except Exception as e:
        print(f"[!] Download failed: {e}")
        raise

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Download official Meta SAM2 model checkpoints.")
    parser.add_argument(
        "--model",
        type=str,
        default="sam2_hiera_small",
        choices=list(CHECKPOINT_URLS.keys()),
        help="SAM2 model checkpoint variant to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Target directory for saving downloaded weights.",
    )

    args = parser.parse_args()
    download_checkpoint(model_name=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
