"""
Download DINOv2 ViT-B14 checkpoint from Facebook's torch hub.

Usage: python setup_model.py

We use DINOv2 ViT-B14 (with registers) as the feature extractor.
DINOv3 / MedDINOv3 can be swapped in later as a drop-in upgrade.
"""

import os
import torch


def setup():
    os.makedirs("weights", exist_ok=True)

    print("Downloading DINOv2 ViT-B14 (with registers) via torch.hub...")
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True
        )
    except Exception as e:
        print(f"torch.hub.load with registers failed: {e}")
        print("Trying DINOv2 ViT-B14 without registers...")
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
        except Exception as e2:
            print(f"torch.hub.load without registers also failed: {e2}")
            print(
                "Please download the checkpoint manually from "
                "https://huggingface.co/facebook/dinov2-base "
                "and place it as weights/dinov2_vitb14.pth"
            )
            return

    torch.save(model.state_dict(), "weights/dinov2_vitb14.pth")
    print("Checkpoint saved to weights/dinov2_vitb14.pth")

    # Check device availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU): available")
    else:
        print("MPS not available - will use CPU")


if __name__ == "__main__":
    setup()
