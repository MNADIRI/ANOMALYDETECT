"""
Anomaly scoring: direct nearest-neighbor cosine distance (AnomalyDINO paradigm)
with 3×3 spatial tolerance and foreground masking.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


# ---------------------------------------------------------------------------
# Foreground mask — explicit block-mean HU computation
# ---------------------------------------------------------------------------

def _create_foreground_mask(
    volume_hu: np.ndarray,
    grid_shape: tuple[int, int, int],
    hu_threshold: float = -200.0,
) -> np.ndarray:
    """
    Create foreground mask by computing mean HU per patch region.

    Maps each feature grid cell to a block of voxels in the HU volume
    and computes the mean HU. Foreground = mean HU > threshold.

    Parameters
    ----------
    volume_hu : [D_hu, H, W] float32
    grid_shape : (D_feat, Hp, Wp) — feature grid dimensions
    hu_threshold : patches with mean HU below this are background

    Returns
    -------
    mask : [D_feat, Hp, Wp] bool
    """
    D_feat, Hp, Wp = grid_shape
    D_hu, H_hu, W_hu = volume_hu.shape

    slice_ratio = D_hu / D_feat
    patch_h = H_hu / Hp
    patch_w = W_hu / Wp

    mask = np.zeros((D_feat, Hp, Wp), dtype=bool)

    for d in range(D_feat):
        d_start = int(d * slice_ratio)
        d_end = max(d_start + 1, min(int((d + 1) * slice_ratio), D_hu))
        for i in range(Hp):
            h_s = int(i * patch_h)
            h_e = max(h_s + 1, min(int((i + 1) * patch_h), H_hu))
            for j in range(Wp):
                w_s = int(j * patch_w)
                w_e = max(w_s + 1, min(int((j + 1) * patch_w), W_hu))
                mask[d, i, j] = volume_hu[d_start:d_end, h_s:h_e, w_s:w_e].mean() > hu_threshold

    return mask


# ---------------------------------------------------------------------------
# Scoring: direct cosine distance with 3×3 spatial tolerance
# ---------------------------------------------------------------------------

def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute change scores using direct cosine distance (AnomalyDINO paradigm).

    For each patch in the new scan, find the best matching patch in the
    reference scan within a 3×3 spatial neighborhood, and use cosine
    distance as the anomaly score.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] — new scan features (L2-normalized, C=2304)
    features_ref : [D, Hp, Wp, C] — reference features (L2-normalized, C=2304)
    volume_hu_new : [D_hu, H, W] — HU volume for foreground masking
    patch_size : ViT patch size (unused, kept for API compat)

    Returns
    -------
    scores : [D, Hp, Wp] float32 — raw cosine distance scores
    """
    D, Hp, Wp, C = features_new.shape
    eps = 1e-8

    # 1. L2-normalize (should already be, but safety)
    new_norm = features_new / (np.linalg.norm(features_new, axis=-1, keepdims=True) + eps)
    ref_norm = features_ref / (np.linalg.norm(features_ref, axis=-1, keepdims=True) + eps)

    # 2. Cosine distance with 3×3 spatial tolerance
    ref_padded = np.pad(ref_norm, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
    best_sim = np.full((D, Hp, Wp), -1.0, dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            candidate = ref_padded[:, di:di + Hp, dj:dj + Wp, :]
            sim = np.sum(new_norm * candidate, axis=-1)
            best_sim = np.maximum(best_sim, sim)

    distances = 1.0 - best_sim
    distances = np.maximum(distances, 0.0)

    # 3. Foreground mask
    fg_mask = None
    if volume_hu_new is not None:
        fg_mask = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg = fg_mask.sum()
        print(f"  Foreground mask: {n_fg}/{fg_mask.size} patches "
              f"({100 * n_fg / fg_mask.size:.1f}%)")

    # 4. Light Gaussian smoothing
    distances = gaussian_filter(
        distances.astype(np.float64),
        sigma=[0.3, 0.7, 0.7],
    ).astype(np.float32)

    # 5. Mask background
    if fg_mask is not None:
        distances[~fg_mask] = 0.0

    # 6. Debug stats
    if fg_mask is not None and fg_mask.any():
        fg_d = distances[fg_mask]
        print(f"  Cosine distances (fg): min={fg_d.min():.4f}, max={fg_d.max():.4f}, "
              f"mean={fg_d.mean():.4f}, "
              f"p95={np.percentile(fg_d, 95):.4f}, "
              f"p99={np.percentile(fg_d, 99):.4f}")
    else:
        print(f"  Distances: min={distances.min():.4f}, max={distances.max():.4f}")

    return distances


def upsample_scores(
    scores: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Upsample scores to target resolution using bilinear interpolation."""
    factors = tuple(t / s for t, s in zip(target_shape, scores.shape))
    return zoom(scores, factors, order=1).astype(np.float32)
