"""
Anomaly scoring: PCA whitening on reference + direct cosine distance
with 3x3 spatial tolerance + z-score normalization.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA with whitening — fitted on reference features
# ---------------------------------------------------------------------------

def fit_and_apply_pca(
    feat_ref: np.ndarray,
    feat_new: np.ndarray,
    fg_mask: np.ndarray | None = None,
    n_components: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA (whitened) on reference features, transform both volumes.

    Parameters
    ----------
    feat_ref : [D, Hp, Wp, 768] — reference patch features (L2-normalized)
    feat_new : [D, Hp, Wp, 768] — new scan patch features (L2-normalized)
    fg_mask : [D, Hp, Wp] bool — foreground mask (optional, for fitting)
    n_components : PCA dimensionality (64 is sufficient for anatomical variance)

    Returns
    -------
    ref_pca : [D, Hp, Wp, n_components] float32
    new_pca : [D, Hp, Wp, n_components] float32
    """
    D, Hp, Wp, C = feat_ref.shape

    # Collect reference patches for PCA fitting
    if fg_mask is not None and fg_mask.any():
        ref_flat = feat_ref[fg_mask]  # [N_fg, 768]
    else:
        ref_flat = feat_ref.reshape(-1, C)

    # Subsample to 50000 for speed
    max_samples = 50_000
    if ref_flat.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(ref_flat.shape[0], max_samples, replace=False)
        ref_sample = ref_flat[idx]
    else:
        ref_sample = ref_flat

    # Fit PCA with whitening on reference
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(ref_sample)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components, whiten=True, "
          f"variance explained: {variance_explained:.1%}")

    # Transform both volumes
    ref_all = feat_ref.reshape(-1, C)
    new_all = feat_new.reshape(-1, C)

    ref_pca = pca.transform(ref_all).reshape(D, Hp, Wp, n_components).astype(np.float32)
    new_pca = pca.transform(new_all).reshape(D, Hp, Wp, n_components).astype(np.float32)

    return ref_pca, new_pca


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
# Scoring: direct cosine distance with 3x3 spatial tolerance + z-score
# ---------------------------------------------------------------------------

def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute change scores using direct cosine distance in PCA space
    with 3x3 spatial tolerance and z-score normalization.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] — new scan features (PCA-transformed)
    features_ref : [D, Hp, Wp, C] — reference features (PCA-transformed)
    volume_hu_new : [D_hu, H, W] — HU volume for foreground masking
    patch_size : ViT patch size (unused, kept for API compatibility)

    Returns
    -------
    z_scores : [D, Hp, Wp] float32 — z-score normalized change map
    """
    D, Hp, Wp, C = features_new.shape
    eps = 1e-8

    # 1. L2-normalize
    new_norm = features_new / (np.linalg.norm(features_new, axis=-1, keepdims=True) + eps)
    ref_norm = features_ref / (np.linalg.norm(features_ref, axis=-1, keepdims=True) + eps)

    # 2. Cosine distance with 3x3 spatial tolerance (vectorized)
    #    Pad reference, compute similarity for all 9 neighbor offsets, take max
    ref_padded = np.pad(ref_norm, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')

    best_sim = np.full((D, Hp, Wp), -1.0, dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            candidate = ref_padded[:, di:di + Hp, dj:dj + Wp, :]  # [D, Hp, Wp, C]
            sim = np.sum(new_norm * candidate, axis=-1)  # [D, Hp, Wp]
            best_sim = np.maximum(best_sim, sim)

    distances = 1.0 - best_sim  # [D, Hp, Wp], range [0, 2]
    distances = np.maximum(distances, 0.0)

    # 3. Foreground mask
    fg_mask = None
    if volume_hu_new is not None:
        fg_mask = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg = fg_mask.sum()
        print(f"  Foreground mask: {n_fg}/{fg_mask.size} patches "
              f"({100 * n_fg / fg_mask.size:.1f}%)")

    # 4. Z-score normalization (median + MAD) on foreground
    if fg_mask is not None and fg_mask.any():
        fg_dists = distances[fg_mask]
    else:
        fg_dists = distances.ravel()

    median = np.median(fg_dists)
    mad = np.median(np.abs(fg_dists - median))
    mad = max(mad, 1e-6)  # minimal floor — only prevent division by zero

    z_scores = (distances - median) / (1.4826 * mad)
    z_scores = np.maximum(z_scores, 0.0)  # only positive z-scores

    # 5. Gaussian smoothing
    z_scores = gaussian_filter(
        z_scores.astype(np.float64),
        sigma=[0.5, 1.0, 1.0],
    ).astype(np.float32)

    # 6. Mask background
    if fg_mask is not None:
        z_scores[~fg_mask] = 0.0

    # Debug stats
    if fg_mask is not None and fg_mask.any():
        fg_z = z_scores[fg_mask]
        print(f"  Z-scores (fg): median_dist={median:.6f}, MAD={mad:.6f}")
        print(f"  Z-scores (fg): min={fg_z.min():.2f}, max={fg_z.max():.2f}, "
              f"mean={fg_z.mean():.2f}, "
              f"p95={np.percentile(fg_z, 95):.2f}, "
              f"p99={np.percentile(fg_z, 99):.2f}")
        print(f"  Above z>3: {(fg_z > 3).sum()}, z>5: {(fg_z > 5).sum()}")
    else:
        print(f"  Z-scores: min={z_scores.min():.2f}, max={z_scores.max():.2f}")

    return z_scores


def upsample_scores(
    scores: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Upsample scores to target resolution using bilinear interpolation."""
    factors = tuple(t / s for t, s in zip(target_shape, scores.shape))
    return zoom(scores, factors, order=1).astype(np.float32)
