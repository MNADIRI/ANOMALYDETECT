"""
Anomaly scoring: cosine distance between patch features,
robust z-score normalisation, and spatial smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
) -> np.ndarray:
    """
    Compute a volume of z-scores indicating change magnitude.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] – new scan features
    features_ref : [D, Hp, Wp, C] – registered reference features

    Returns
    -------
    z_scores : [D, Hp, Wp] float32
    """
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

    # 4. Robust z-score: (x - median) / (MAD * 1.4826)
    median = np.median(distance)
    mad = np.median(np.abs(distance - median))
    if mad < eps:
        mad = np.std(distance)
    z_scores = (distance - median) / (mad * 1.4826 + eps)

    # 5. Spatial smoothing
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
