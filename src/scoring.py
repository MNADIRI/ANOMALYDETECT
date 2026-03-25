"""
Anomaly scoring: cosine distance between patch features,
robust z-score normalisation, and spatial smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _create_body_mask(
    volume_hu: np.ndarray,
    patch_size: int,
    hu_threshold: float = -900.0,
    min_tissue_fraction: float = 0.1,
) -> np.ndarray:
    """
    Create a patch-level mask that excludes air regions.

    A patch is considered "body" if at least `min_tissue_fraction` of its
    pixels have HU > hu_threshold.

    Parameters
    ----------
    volume_hu : [D, H, W] float32
    patch_size : size of each ViT patch (e.g. 14)
    hu_threshold : HU value below which a pixel is considered air
    min_tissue_fraction : minimum fraction of non-air pixels in a patch

    Returns
    -------
    mask : [D, Hp, Wp] bool — True = body, False = air
    """
    D, H, W = volume_hu.shape
    Hp = H // patch_size
    Wp = W // patch_size

    # Crop to exact multiple of patch_size
    vol = volume_hu[:, :Hp * patch_size, :Wp * patch_size]

    # Reshape into patches and compute tissue fraction
    vol = vol.reshape(D, Hp, patch_size, Wp, patch_size)
    tissue = (vol > hu_threshold).astype(np.float32)
    fraction = tissue.mean(axis=(2, 4))  # [D, Hp, Wp]

    return fraction >= min_tissue_fraction


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
    patch_size : ViT patch size for mask computation

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

    # 4. Create body mask to exclude air regions
    body_mask = None
    if volume_hu_new is not None:
        body_mask = _create_body_mask(volume_hu_new, patch_size)
        # Ensure mask shape matches feature grid
        if body_mask.shape != (D, Hp, Wp):
            body_mask = zoom(
                body_mask.astype(np.float32),
                (D / body_mask.shape[0], Hp / body_mask.shape[1], Wp / body_mask.shape[2]),
                order=0,
            ) > 0.5

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

    # 6. Zero out air regions
    if body_mask is not None:
        z_scores[~body_mask] = 0.0

    # 7. Clamp negative z-scores (we only care about increases)
    z_scores = np.maximum(z_scores, 0.0)

    # 8. Spatial smoothing
    z_scores = gaussian_filter(
        z_scores.astype(np.float64),
        sigma=[1.0, 1.5, 1.5],  # (z, y, x) in patches
    ).astype(np.float32)

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
