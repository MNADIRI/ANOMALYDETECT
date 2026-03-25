"""
Feature extraction using DINOv2 ViT-B14 (frozen).

Extracts patch-token activations from the last transformer block
and CLS token embeddings for slice matching.
"""

import math
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ImageNet normalization constants (required for DINOv2)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


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


class LayerScale(nn.Module):
    """Per-channel learnable scaling (used in DINOv2)."""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    """Standard transformer block with Mlp FFN."""
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN matching DINOv2's exact implementation."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        swiglu_hidden = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * swiglu_hidden)
        self.w3 = nn.Linear(swiglu_hidden, in_features)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(nn.functional.silu(x1) * x2)


class BlockWithSwiGLU(nn.Module):
    """Transformer block with SwiGLU FFN."""
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_value=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLUFFN(dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
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
        init_values=1e-5,
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
            block_cls(embed_dim, num_heads, mlp_ratio, init_values=init_values)
            for _ in range(depth)
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
            patch_pos.float(), size=(h0, w0), mode="bicubic", align_corners=False
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
    has_swiglu = any(k.endswith(".mlp.w12.weight") for k in state_dict.keys())
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
    """
    device = get_device()
    patch_size = 14

    weight_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "weights", "dinov2_vitb14.pth"
    )

    if os.path.exists(weight_path):
        print(f"Loading DINOv2 from local checkpoint: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

        use_swiglu, n_register = _detect_checkpoint_type(state_dict)
        print(f"  Detected: SwiGLU={use_swiglu}, register_tokens={n_register}")

        # Debug key structure
        ckpt_keys = sorted(state_dict.keys())
        print(f"  Checkpoint: {len(ckpt_keys)} keys")
        mlp_keys = [k for k in ckpt_keys if "blocks.0.mlp" in k]
        print(f"  blocks.0.mlp keys: {mlp_keys}")

        model = DinoVisionTransformer(
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_register_tokens=n_register,
            use_swiglu=use_swiglu,
        )

        model_keys = sorted(model.state_dict().keys())
        print(f"  Model: {len(model_keys)} keys")
        model_mlp_keys = [k for k in model_keys if "blocks.0.mlp" in k]
        print(f"  Model blocks.0.mlp keys: {model_mlp_keys}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing_crit = [k for k in missing if "mask_token" not in k and "head" not in k]
        unexpected_crit = [k for k in unexpected if "mask_token" not in k and "head" not in k]

        if missing_crit:
            print(f"  MISSING keys ({len(missing_crit)}): {missing_crit[:10]}")
        if unexpected_crit:
            print(f"  UNEXPECTED keys ({len(unexpected_crit)}): {unexpected_crit[:10]}")
        if not missing_crit and not unexpected_crit:
            print("  All weights loaded successfully!")

        model = model.to(device)
        model.eval()
        print(f"  Model loaded on {device}")
        return model, device, patch_size, n_register

    # Fallback: torch.hub
    print("Local checkpoint not found, trying torch.hub...")
    for hub_name, n_reg in [("dinov2_vitb14_reg", 4), ("dinov2_vitb14", 0)]:
        try:
            model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
            n_register = getattr(model, "num_register_tokens", n_reg)
            patch_size = getattr(model, "patch_size", 14)
            model = model.to(device)
            model.eval()
            return model, device, patch_size, n_register
        except Exception:
            continue

    raise RuntimeError("Could not load DINOv2. Run setup_model.py first.")


# ---------------------------------------------------------------------------
# Feature extraction — single last layer (DINO-AD approach)
# ---------------------------------------------------------------------------

EXTRACT_LAYER = 11  # Last transformer block
INPUT_SIZE = 512


@torch.no_grad()
def extract_features(
    model: nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: int = 14,
    n_register: int = 4,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract last-layer patch features and CLS token for every slice.

    Parameters
    ----------
    volume : [D, 3, H, W] float32, values in [0, 1]

    Returns
    -------
    patch_features : [D, Hp, Wp, 768]
    cls_tokens : [D, 768] — for slice matching
    """
    D, C, H, W = volume.shape

    # Pad to multiple of patch_size
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
    embed_dim = 768

    # Precompute normalization tensors on device
    img_mean = IMAGENET_MEAN.to(device)
    img_std = IMAGENET_STD.to(device)

    all_features = np.empty((D, Hp, Wp, embed_dim), dtype=np.float32)
    all_cls = np.empty((D, embed_dim), dtype=np.float32)

    for i in range(D):
        # Prepare single-slice batch [1, 3, H, W]
        slice_tensor = torch.from_numpy(volume[i : i + 1]).to(device)

        # ImageNet normalization
        slice_tensor = (slice_tensor - img_mean) / img_std

        # Forward pass — get final output (after norm)
        output = model(slice_tensor)  # [1, 1+n_reg+Hp*Wp, 768]

        # Extract CLS token (always first)
        cls_token = output[:, 0, :]  # [1, 768]
        all_cls[i] = cls_token.cpu().numpy()

        # Extract patch tokens (skip CLS + register tokens)
        n_skip = 1 + n_register
        patch_tokens = output[:, n_skip:, :]  # [1, Hp*Wp, 768]

        # Auto-detect if token count doesn't match
        expected = Hp * Wp
        if patch_tokens.shape[1] != expected:
            n_skip_auto = output.shape[1] - expected
            if n_skip_auto > 0:
                patch_tokens = output[:, n_skip_auto:, :]

        patch_tokens = patch_tokens[:, :expected, :]
        patch_tokens = patch_tokens.reshape(1, Hp, Wp, embed_dim)
        all_features[i] = patch_tokens[0].cpu().numpy()

        # Memory cleanup
        del slice_tensor, output
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        if progress_callback is not None:
            progress_callback((i + 1) / D)

    return all_features, all_cls


def match_slices(
    cls_new: np.ndarray,
    cls_ref: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    For each new slice, find the best-matching reference slice
    within a local window using CLS token cosine similarity.

    Parameters
    ----------
    cls_new : [D_new, 768]
    cls_ref : [D_ref, 768]
    window : search window (±window slices around positional match)

    Returns
    -------
    indices : [D_new] — index into reference for each new slice
    """
    D_new = cls_new.shape[0]
    D_ref = cls_ref.shape[0]

    # L2-normalize
    eps = 1e-8
    new_norm = cls_new / (np.linalg.norm(cls_new, axis=-1, keepdims=True) + eps)
    ref_norm = cls_ref / (np.linalg.norm(cls_ref, axis=-1, keepdims=True) + eps)

    indices = np.zeros(D_new, dtype=np.int64)

    for i in range(D_new):
        # Positional match
        pos_match = int(i * D_ref / D_new)
        start = max(0, pos_match - window)
        end = min(D_ref, pos_match + window + 1)

        # Cosine similarity with candidates
        sims = ref_norm[start:end] @ new_norm[i]
        best_local = np.argmax(sims)
        indices[i] = start + best_local

    return indices
