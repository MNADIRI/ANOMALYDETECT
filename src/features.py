"""
Feature extraction using DINOv2 ViT-B14 (frozen, multi-layer).

Extracts intermediate patch-token activations from layers [2, 5, 8, 11],
concatenates them, and optionally reduces dimensionality with PCA.
"""

import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> tuple[nn.Module, torch.device, int, int]:
    """
    Load DINOv2 ViT-B14 and return (model, device, patch_size, n_register_tokens).

    Tries torch.hub first (with registers, then without).
    Falls back to loading a local checkpoint.
    """
    device = get_device()

    # Try hub: with registers
    model = None
    patch_size = 14
    n_register = 4

    weight_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "weights", "dinov2_vitb14.pth"
    )

    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True
        )
        n_register = getattr(model, "num_register_tokens", 4)
        patch_size = getattr(model, "patch_size", 14)
    except Exception:
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
            n_register = getattr(model, "num_register_tokens", 0)
            patch_size = getattr(model, "patch_size", 14)
        except Exception:
            pass

    # Fall back to local weights
    if model is None and os.path.exists(weight_path):
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False
            )
            state = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            n_register = getattr(model, "num_register_tokens", 0)
            patch_size = getattr(model, "patch_size", 14)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load DINOv2 model from hub or local weights: {exc}"
            ) from exc

    if model is None:
        raise RuntimeError(
            "Could not load DINOv2. Run setup_model.py first or ensure "
            "internet access for torch.hub."
        )

    model = model.to(device)
    model.eval()
    return model, device, patch_size, n_register


# ---------------------------------------------------------------------------
# Feature extractor with forward hooks
# ---------------------------------------------------------------------------

class MultiLayerFeatureExtractor:
    """Register forward hooks on specified transformer blocks."""

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.features: dict[int, torch.Tensor] = {}
        self._hooks = []

        # DINOv2 stores blocks in model.blocks
        blocks = model.blocks
        for idx in layer_indices:
            hook = blocks[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook_fn(_module, _input, output):
            self.features[idx] = output
        return hook_fn

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

EXTRACT_LAYERS = [2, 5, 8, 11]
INPUT_SIZE = 512


@torch.no_grad()
def extract_features(
    model: nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: int = 14,
    n_register: int = 4,
    extract_layers: list[int] | None = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """
    Extract multi-layer patch features for every slice in the volume.

    Parameters
    ----------
    volume : [D, 3, H, W] float32, values in [0, 1]
    patch_size : ViT patch size (14 for DINOv2)
    n_register : number of register tokens to skip

    Returns
    -------
    features : [D, Hp, Wp, C]  where C = len(extract_layers) * 768
    """
    if extract_layers is None:
        extract_layers = EXTRACT_LAYERS

    D, C, H, W = volume.shape

    # Ensure H, W are multiples of patch_size by padding
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        volume = np.pad(
            volume,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
        _, _, H, W = volume.shape

    Hp = H // patch_size
    Wp = W // patch_size
    embed_dim = 768  # ViT-B
    feat_dim = len(extract_layers) * embed_dim

    extractor = MultiLayerFeatureExtractor(model, extract_layers)

    all_features = np.empty((D, Hp, Wp, feat_dim), dtype=np.float32)

    for i in range(D):
        extractor.clear()

        # Prepare single-slice batch [1, 3, H, W]
        slice_tensor = torch.from_numpy(volume[i : i + 1]).to(device)

        # Forward pass
        _ = model(slice_tensor)

        # Gather features from hooks
        layer_feats = []
        for layer_idx in extract_layers:
            tokens = extractor.features[layer_idx]  # [1, L, 768]
            # Remove CLS + register tokens
            n_skip = 1 + n_register
            patch_tokens = tokens[:, n_skip:, :]  # [1, Hp*Wp, 768]

            # Verify count
            expected = Hp * Wp
            actual = patch_tokens.shape[1]
            if actual != expected:
                # Auto-detect register count
                n_skip_auto = tokens.shape[1] - expected
                if n_skip_auto > 0:
                    patch_tokens = tokens[:, n_skip_auto:, :]

            patch_tokens = patch_tokens.reshape(1, Hp, Wp, embed_dim)
            layer_feats.append(patch_tokens.cpu())

        # Concatenate layers: [1, Hp, Wp, feat_dim]
        combined = torch.cat(layer_feats, dim=-1)
        all_features[i] = combined[0].numpy()

        # Memory cleanup
        del slice_tensor
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        if progress_callback is not None:
            progress_callback((i + 1) / D)

    extractor.remove_hooks()
    return all_features


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_features(
    features_ref: np.ndarray,
    features_new: np.ndarray,
    n_components: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on reference features, apply to both volumes.

    Parameters
    ----------
    features_ref, features_new : [D, Hp, Wp, C]

    Returns
    -------
    ref_reduced, new_reduced : [D, Hp, Wp, n_components]
    """
    D, Hp, Wp, C = features_ref.shape

    # Flatten to (N, C)
    ref_flat = features_ref.reshape(-1, C)
    new_flat = features_new.reshape(-1, C)

    # Subsample for fitting PCA (max 50000 patches)
    n_total = ref_flat.shape[0]
    max_samples = 50_000
    if n_total > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, max_samples, replace=False)
        fit_data = ref_flat[indices]
    else:
        fit_data = ref_flat

    n_components = min(n_components, C, fit_data.shape[0])
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(fit_data)

    ref_reduced = pca.transform(ref_flat).reshape(D, Hp, Wp, n_components)
    new_reduced = pca.transform(new_flat).reshape(
        features_new.shape[0], Hp, Wp, n_components
    )

    return ref_reduced.astype(np.float32), new_reduced.astype(np.float32)
