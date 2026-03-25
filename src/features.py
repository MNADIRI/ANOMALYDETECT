"""
Feature extraction using DINOv2 ViT-B14 (frozen, multi-layer).

Extracts intermediate patch-token activations from layers [2, 5, 8, 11],
concatenates them, and optionally reduces dimensionality with PCA.
"""

import math
import os
from functools import partial
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
# Minimal DINOv2 ViT-B architecture (no external dependency)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN used in DINOv2."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        # DINOv2 uses 2/3 * 4 * dim for SwiGLU hidden
        swiglue_hidden = int(hidden_features * 2 / 3)
        # Round to multiple of 256 (as in DINOv2)
        swiglue_hidden = (swiglue_hidden + 255) // 256 * 256
        self.w12 = nn.Linear(in_features, 2 * swiglue_hidden)
        self.w3 = nn.Linear(swiglue_hidden, in_features)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(nn.functional.silu(x1) * x2)


class BlockWithSwiGLU(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLUFFN(dim, hidden_features=int(dim * mlp_ratio))
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DinoVisionTransformer(nn.Module):
    """Minimal DINOv2 ViT-B14 that matches the checkpoint structure."""

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=0,
        use_swiglu=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim)
            )
        else:
            self.register_tokens = None

        block_cls = BlockWithSwiGLU if use_swiglu else Block
        self.blocks = nn.ModuleList([
            block_cls(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N:
            return self.pos_embed
        class_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=(h0, w0), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([class_pos, patch_pos], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)

        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


def _detect_checkpoint_type(state_dict: dict) -> tuple[bool, int]:
    """Detect if checkpoint uses SwiGLU and how many register tokens."""
    has_swiglu = any("w12" in k for k in state_dict.keys())
    has_registers = "register_tokens" in state_dict
    n_reg = 0
    if has_registers:
        n_reg = state_dict["register_tokens"].shape[1]
    return has_swiglu, n_reg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> tuple[nn.Module, torch.device, int, int]:
    """
    Load DINOv2 ViT-B14 and return (model, device, patch_size, n_register_tokens).

    Loads from local checkpoint (weights/dinov2_vitb14.pth).
    Falls back to torch.hub if checkpoint not found.
    """
    device = get_device()
    patch_size = 14

    weight_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "weights", "dinov2_vitb14.pth"
    )

    # Strategy 1: Load from local checkpoint (no internet needed)
    if os.path.exists(weight_path):
        print(f"Loading DINOv2 from local checkpoint: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

        # Detect architecture from checkpoint keys
        use_swiglu, n_register = _detect_checkpoint_type(state_dict)
        print(f"  SwiGLU: {use_swiglu}, register tokens: {n_register}")

        model = DinoVisionTransformer(
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_register_tokens=n_register,
            use_swiglu=use_swiglu,
        )

        # Load weights (strict=False to handle minor mismatches like mask_token)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out non-critical missing keys
            critical = [k for k in missing if "mask_token" not in k]
            if critical:
                print(f"  Warning: missing keys: {critical[:5]}...")
        if unexpected:
            print(f"  Warning: unexpected keys: {unexpected[:5]}...")

        model = model.to(device)
        model.eval()
        print(f"  Model loaded on {device}")
        return model, device, patch_size, n_register

    # Strategy 2: Try torch.hub (needs internet)
    print("Local checkpoint not found, trying torch.hub...")
    for hub_name, n_reg in [("dinov2_vitb14_reg", 4), ("dinov2_vitb14", 0)]:
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", hub_name, pretrained=True
            )
            n_register = getattr(model, "num_register_tokens", n_reg)
            patch_size = getattr(model, "patch_size", 14)
            model = model.to(device)
            model.eval()
            return model, device, patch_size, n_register
        except Exception:
            continue

    raise RuntimeError(
        "Could not load DINOv2. Run setup_model.py first to download "
        "the checkpoint, or ensure internet access for torch.hub."
    )


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
