"""
Anomaly scoring: PCA reconstruction error on reference-fitted subspace
with z-score normalization.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA with whitening — fitted on reference features
# ---------------------------------------------------------------------------

def fit_pca(
    feat_ref: np.ndarray,
    fg_mask: np.ndarray | None = None,
    n_components: int = 64,
) -> PCA:
    """
    Fit PCA (whitened) on reference features only.

    Parameters
    ----------
    feat_ref : [D, Hp, Wp, 768] — reference patch features (L2-normalized)
    fg_mask : [D, Hp, Wp] bool — foreground mask (optional, for fitting)
    n_components : PCA dimensionality

    Returns
    -------
    pca : fitted PCA model
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
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(ref_sample)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components, whiten=False, "
          f"variance explained: {variance_explained:.1%}")

    return pca


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
# Scoring: PCA reconstruction error + z-score normalization
# ---------------------------------------------------------------------------

def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    pca: PCA,
    volume_hu_new: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute change scores using PCA reconstruction error.

    The PCA was fitted on reference features. Anomalous regions in the new
    scan will have high reconstruction error because they don't fit the
    reference subspace.

    Z-score normalization uses the reference reconstruction error distribution
    as baseline.

    Parameters
    ----------
    features_new : [D, Hp, Wp, 768] — new scan features (original, NOT PCA-transformed)
    features_ref : [D, Hp, Wp, 768] — reference features (original)
    pca : fitted PCA model (from fit_pca)
    volume_hu_new : [D_hu, H, W] — HU volume for foreground masking
    patch_size : ViT patch size (unused, kept for API compatibility)

    Returns
    -------
    z_scores : [D, Hp, Wp] float32 — z-score normalized change map
    """
    D, Hp, Wp, C = features_new.shape

    # 1. Compute reconstruction error for both volumes
    new_flat = features_new.reshape(-1, C)
    ref_flat = features_ref.reshape(-1, C)

    new_projected = pca.transform(new_flat)
    new_reconstructed = pca.inverse_transform(new_projected)
    recon_error_new = np.linalg.norm(new_flat - new_reconstructed, axis=-1)
    recon_error_new = recon_error_new.reshape(D, Hp, Wp).astype(np.float32)

    ref_projected = pca.transform(ref_flat)
    ref_reconstructed = pca.inverse_transform(ref_projected)
    recon_error_ref = np.linalg.norm(ref_flat - ref_reconstructed, axis=-1)
    recon_error_ref = recon_error_ref.reshape(D, Hp, Wp).astype(np.float32)

    print(f"  Reconstruction error — new: mean={recon_error_new.mean():.6f}, "
          f"max={recon_error_new.max():.6f}")
    print(f"  Reconstruction error — ref: mean={recon_error_ref.mean():.6f}, "
          f"max={recon_error_ref.max():.6f}")

    # 2. Foreground mask
    fg_mask = None
    if volume_hu_new is not None:
        fg_mask = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg = fg_mask.sum()
        print(f"  Foreground mask: {n_fg}/{fg_mask.size} patches "
              f"({100 * n_fg / fg_mask.size:.1f}%)")

    # 3. Z-score normalization using REFERENCE reconstruction error as baseline
    #    This way, normal anatomy that reconstructs similarly in both scans
    #    gets low z-scores, while hemorrhage (high recon error in new, low in ref)
    #    gets high z-scores.
    if fg_mask is not None and fg_mask.any():
        ref_fg_errors = recon_error_ref[fg_mask]
    else:
        ref_fg_errors = recon_error_ref.ravel()

    median_ref = np.median(ref_fg_errors)
    mad_ref = np.median(np.abs(ref_fg_errors - median_ref))
    mad_ref = max(mad_ref, 1e-6)  # minimal floor — only prevent division by zero

    z_scores = (recon_error_new - median_ref) / (1.4826 * mad_ref)
    z_scores = np.maximum(z_scores, 0.0)  # only positive z-scores

    # 4. Gaussian smoothing
    z_scores = gaussian_filter(
        z_scores.astype(np.float64),
        sigma=[0.5, 1.0, 1.0],
    ).astype(np.float32)

    # 5. Mask background
    if fg_mask is not None:
        z_scores[~fg_mask] = 0.0

    # Debug stats
    if fg_mask is not None and fg_mask.any():
        fg_z = z_scores[fg_mask]
        print(f"  Z-scores (fg): median_ref_error={median_ref:.6f}, MAD={mad_ref:.6f}")
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
