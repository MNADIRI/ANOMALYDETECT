"""
Anomaly scoring: cosine distance between patch features,
robust z-score normalisation, and spatial smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _create_body_mask_at_grid(
    volume_hu: np.ndarray,
    grid_shape: tuple[int, int, int],
    hu_threshold: float = -900.0,
    min_tissue_fraction: float = 0.1,
) -> np.ndarray:
    """
    Create a body mask at the same resolution as the feature grid.

    Parameters
    ----------
    volume_hu : [D, H, W] float32
    grid_shape : (D_feat, Hp, Wp) — the feature grid dimensions
    hu_threshold : HU below which a pixel is air
    min_tissue_fraction : min fraction of non-air per patch

    Returns
    -------
    mask : [D_feat, Hp, Wp] bool
    """
    D_feat, Hp, Wp = grid_shape
    D_hu, H_hu, W_hu = volume_hu.shape

    # Compute effective patch size from HU volume to grid
    patch_h = H_hu / Hp
    patch_w = W_hu / Wp
    slice_ratio = D_hu / D_feat

    mask = np.zeros((D_feat, Hp, Wp), dtype=bool)

    for d in range(D_feat):
        # Map grid slice to HU slices
        d_start = int(d * slice_ratio)
        d_end = min(int((d + 1) * slice_ratio), D_hu)
        if d_end <= d_start:
            d_end = d_start + 1

        for hp in range(Hp):
            h_start = int(hp * patch_h)
            h_end = min(int((hp + 1) * patch_h), H_hu)
            for wp in range(Wp):
                w_start = int(wp * patch_w)
                w_end = min(int((wp + 1) * patch_w), W_hu)

                patch = volume_hu[d_start:d_end, h_start:h_end, w_start:w_end]
                if patch.size > 0:
                    tissue_frac = (patch > hu_threshold).mean()
                    mask[d, hp, wp] = tissue_frac >= min_tissue_fraction

    return mask


def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute a volume of z-scores indicating change magnitude.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] – new scan features
    features_ref : [D, Hp, Wp, C] – registered reference features
    volume_hu_new : [D, H, W] float32 – HU volume for body masking (optional)
    patch_size : ViT patch size (unused now, kept for API compat)

    Returns
    -------
    z_scores : [D, Hp, Wp] float32
    """
    D, Hp, Wp, C = features_new.shape

    # 1. L2-normalise each feature vector
    eps = 1e-8
    ref_norm = features_ref / (
        np.linalg.norm(features_ref, axis=-1, keepdims=True) + eps
    )
    new_norm = features_new / (
        np.linalg.norm(features_new, axis=-1, keepdims=True) + eps
    )

    # 2. Cosine similarity (dot product of normalised vectors)
    cosine_sim = np.sum(ref_norm * new_norm, axis=-1)  # [D, Hp, Wp]

    # 3. Cosine distance
    distance = 1.0 - cosine_sim  # [D, Hp, Wp], range [0, 2]

    # 4. Create body mask at feature grid resolution
    body_mask = None
    if volume_hu_new is not None:
        body_mask = _create_body_mask_at_grid(
            volume_hu_new, (D, Hp, Wp)
        )

    # 5. Robust z-score on body voxels only
    if body_mask is not None and body_mask.any():
        body_distances = distance[body_mask]
    else:
        body_distances = distance.ravel()

    median = np.median(body_distances)
    mad = np.median(np.abs(body_distances - median))

    # Floor on MAD to prevent z-score explosion with near-identical scans
    mad = max(mad, 0.01)

    z_scores = (distance - median) / (mad * 1.4826)

    # 6. Clamp negative z-scores (we only care about increases)
    z_scores = np.maximum(z_scores, 0.0)

    # 7. Spatial smoothing BEFORE masking air
    #    (avoids air zeros contaminating body edges)
    z_scores = gaussian_filter(
        z_scores.astype(np.float64),
        sigma=[0.5, 1.0, 1.0],  # reduced sigma to preserve focal lesions
    ).astype(np.float32)

    # 8. Zero out air regions AFTER smoothing
    if body_mask is not None:
        z_scores[~body_mask] = 0.0

    # Debug stats
    if body_mask is not None and body_mask.any():
        body_scores = z_scores[body_mask]
        print(f"  z-scores (body only): min={body_scores.min():.2f}, "
              f"max={body_scores.max():.2f}, mean={body_scores.mean():.2f}, "
              f"p95={np.percentile(body_scores, 95):.2f}, "
              f"p99={np.percentile(body_scores, 99):.2f}")
        print(f"  body mask: {body_mask.sum()}/{body_mask.size} voxels "
              f"({100*body_mask.mean():.1f}%)")
    else:
        print(f"  z-scores: min={z_scores.min():.2f}, max={z_scores.max():.2f}, "
              f"mean={z_scores.mean():.2f}")

    return z_scores


def upsample_scores(
    z_scores: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Upsample z-score volume to target resolution using bilinear interpolation.

    Parameters
    ----------
    z_scores : [D_p, Hp, Wp]
    target_shape : (D, H, W) at native resolution

    Returns
    -------
    upsampled : [D, H, W] float32
    """
    factors = tuple(t / s for t, s in zip(target_shape, z_scores.shape))
    return zoom(z_scores, factors, order=1).astype(np.float32)
