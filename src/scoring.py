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
# Simple foreground mask — vectorized mean HU per patch
# ---------------------------------------------------------------------------

def _create_foreground_mask(
    volume_hu: np.ndarray,
    grid_shape: tuple[int, int, int],
    hu_threshold: float = -200.0,
) -> np.ndarray:
    """
    Create foreground mask by computing mean HU per patch region.

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

    # Use zoom to downsample HU volume to grid resolution, then threshold
    factors = (D_feat / D_hu, Hp / H_hu, Wp / W_hu)
    mean_hu = zoom(volume_hu, factors, order=1)

    # Ensure shape matches exactly
    mean_hu = mean_hu[:D_feat, :Hp, :Wp]

    return mean_hu > hu_threshold


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

    # 2. Cosine distance with 3x3 spatial tolerance
    #    For each patch (d, i, j), find the most similar patch in ref within i±1, j±1
    distances = np.zeros((D, Hp, Wp), dtype=np.float32)

    for d in range(D):
        for i in range(Hp):
            i_lo = max(0, i - 1)
            i_hi = min(Hp, i + 2)
            for j in range(Wp):
                j_lo = max(0, j - 1)
                j_hi = min(Wp, j + 2)

                # Query vector
                q = new_norm[d, i, j]  # [C]
                # Candidate neighborhood in reference
                candidates = ref_norm[d, i_lo:i_hi, j_lo:j_hi]  # [<=3, <=3, C]
                # Cosine similarities
                sims = candidates.reshape(-1, C) @ q  # [<=9]
                distances[d, i, j] = 1.0 - sims.max()

    # 3. Foreground mask
    fg_mask = None
    if volume_hu_new is not None:
        fg_mask = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg = fg_mask.sum()
        print(f"  Foreground mask: {n_fg}/{fg_mask.size} patches "
              f"({100*n_fg/fg_mask.size:.1f}%)")

    # 4. Z-score normalization (median + MAD) on foreground
    if fg_mask is not None and fg_mask.any():
        fg_dists = distances[fg_mask]
    else:
        fg_dists = distances.ravel()

    median = np.median(fg_dists)
    mad = np.median(np.abs(fg_dists - median))
    mad = max(mad, 0.01)  # floor to avoid division by zero

    z_scores = (distances - median) / (1.4826 * mad)  # 1.4826 = MAD-to-std scaling
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
        print(f"  Z-scores (fg): median_dist={median:.4f}, MAD={mad:.4f}")
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
