"""
Download DINOv2 ViT-B14 checkpoint.

Usage: python setup_model.py

Downloads directly from Facebook's servers (same URLs torch.hub uses)
without needing the GitHub repo.
"""

import os
import urllib.request
import torch


# Direct download URLs from Facebook (no GitHub dependency)
CHECKPOINT_URLS = [
    # DINOv2 ViT-B14 with registers (preferred)
    (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        "dinov2_vitb14_reg4",
    ),
    # DINOv2 ViT-B14 without registers (fallback)
    (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "dinov2_vitb14",
    ),
]


def download_with_progress(url: str, dest: str):
    """Download a file with a progress indicator."""
    print(f"Downloading from:\n  {url}")
    print(f"Saving to: {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def setup():
    os.makedirs("weights", exist_ok=True)
    dest_path = os.path.join("weights", "dinov2_vitb14.pth")

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"Checkpoint already exists: {dest_path} ({size_mb:.1f} MB)")
        print("Delete it and re-run to download again.")
        return

    for url, name in CHECKPOINT_URLS:
        try:
            print(f"\nTrying {name}...")
            download_with_progress(url, dest_path)

            # Verify the file is a valid checkpoint
            state = torch.load(dest_path, map_location="cpu", weights_only=True)
            n_keys = len(state)
            print(f"Checkpoint valid: {n_keys} parameter tensors")
            print(f"Saved to {dest_path}")

            # Check MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("MPS (Apple Silicon GPU): available")
            else:
                print("MPS not available - will use CPU")
            return

        except Exception as e:
            print(f"Failed with {name}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            continue

    print("\nAll download attempts failed.")
    print("Please download manually:")
    print("  1. Go to https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
    print("  2. Save the file as weights/dinov2_vitb14.pth")


if __name__ == "__main__":
    setup()
