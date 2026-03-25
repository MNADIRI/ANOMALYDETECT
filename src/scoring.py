"""
Anomaly scoring using hybrid DINO-AD methodology:
- Foreground-aware K-means clustering (K=2) on reference features
- Reference self-similarity baseline (delta from baseline)
- Direct patch-to-patch cosine distance
- Multiplicative combination (soft AND gate)
- Percentile normalization, spatial consistency filtering
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_closing, binary_erosion, label


def _create_foreground_mask(
    volume_hu: np.ndarray,
    grid_shape: tuple[int, int, int],
    hu_threshold: float = -900.0,
    min_tissue_fraction: float = 0.25,
) -> np.ndarray:
    """
    Create a foreground mask at feature grid resolution.

    Parameters
    ----------
    volume_hu : [D, H, W] float32
    grid_shape : (D_feat, Hp, Wp)
    hu_threshold : HU below which = air
    min_tissue_fraction : min fraction of tissue per patch

    Returns
    -------
    mask : [D_feat, Hp, Wp] bool
    """
    D_feat, Hp, Wp = grid_shape
    D_hu, H_hu, W_hu = volume_hu.shape

    patch_h = H_hu / Hp
    patch_w = W_hu / Wp
    slice_ratio = D_hu / D_feat

    mask = np.zeros((D_feat, Hp, Wp), dtype=bool)

    for d in range(D_feat):
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

    # Morphological closing to fill small holes, then erosion to remove boundary patches
    for d in range(D_feat):
        mask[d] = binary_closing(mask[d], iterations=2)
        mask[d] = binary_erosion(mask[d], iterations=1)

    return mask


def _percentile_normalize(
    values: np.ndarray,
    fg_mask: np.ndarray | None = None,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Normalize using percentile clipping instead of min-max."""
    eps = 1e-8
    if fg_mask is not None and fg_mask.any():
        fg_vals = values[fg_mask]
    else:
        fg_vals = values.ravel()

    lo = np.percentile(fg_vals, p_low)
    hi = np.percentile(fg_vals, p_high)
    rng = hi - lo
    if rng < eps:
        rng = 1.0

    normalized = (values - lo) / rng
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _spatial_consistency_filter(
    scores: np.ndarray,
    fg_mask: np.ndarray | None,
    min_cluster_size: int = 4,
    suppression_factor: float = 0.2,
) -> np.ndarray:
    """
    Suppress isolated high-scoring patches that don't form coherent clusters.
    Real pathology spans multiple contiguous patches; isolated ones are noise.
    """
    if fg_mask is not None and fg_mask.any():
        fg_scores = scores[fg_mask]
        threshold = np.percentile(fg_scores, 85)
    else:
        threshold = np.percentile(scores[scores > 0], 85) if (scores > 0).any() else 0.5

    binary_map = scores > threshold
    labeled, n_components = label(binary_map)

    if n_components == 0:
        return scores

    filtered = scores.copy()
    for comp_id in range(1, n_components + 1):
        component_mask = labeled == comp_id
        if component_mask.sum() < min_cluster_size:
            filtered[component_mask] *= suppression_factor

    return filtered


def _per_slice_suppression(
    scores: np.ndarray,
    fg_mask: np.ndarray | None,
    slice_anomaly_fraction: float = 0.5,
    dampening_factor: float = 0.3,
) -> np.ndarray:
    """
    Dampen slices where too many patches are anomalous (likely registration error).
    Exception: don't suppress if the slice contains very high scores (genuine large lesion).
    """
    D = scores.shape[0]
    if fg_mask is None:
        return scores

    # Global 75th and 99th percentile
    all_fg = scores[fg_mask] if fg_mask.any() else scores.ravel()
    if all_fg.size == 0:
        return scores
    p75 = np.percentile(all_fg, 75)
    p99 = np.percentile(all_fg, 99)

    filtered = scores.copy()
    for d in range(D):
        slice_fg = fg_mask[d]
        if not slice_fg.any():
            continue

        slice_scores = scores[d][slice_fg]
        fraction_above = (slice_scores > p75).mean()
        slice_max = slice_scores.max()

        # If majority of patches are "anomalous" but no extreme peak → registration issue
        if fraction_above > slice_anomaly_fraction and slice_max < p99:
            filtered[d] *= dampening_factor

    return filtered


def _kmeans_simple(features: np.ndarray, k: int = 2, max_iter: int = 50) -> np.ndarray:
    """
    Simple K-means on feature vectors.

    Parameters
    ----------
    features : [N, C] float32 — L2-normalized feature vectors
    k : number of clusters

    Returns
    -------
    centroids : [K, C] float32
    """
    N, C = features.shape
    if N <= k:
        return features.copy()

    # Initialize with k-means++ style
    rng = np.random.default_rng(42)
    centroids = np.empty((k, C), dtype=np.float32)
    centroids[0] = features[rng.integers(N)]

    for i in range(1, k):
        dists = np.min(
            [np.sum((features - centroids[j]) ** 2, axis=1) for j in range(i)],
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-10)
        centroids[i] = features[rng.choice(N, p=probs)]

    for _ in range(max_iter):
        dists = np.stack([
            np.sum((features - centroids[j]) ** 2, axis=1)
            for j in range(k)
        ], axis=1)
        assignments = np.argmin(dists, axis=1)

        new_centroids = np.empty_like(centroids)
        for j in range(k):
            members = features[assignments == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    patch_size: int = 14,
    n_clusters: int = 2,
) -> np.ndarray:
    """
    Compute anomaly scores using hybrid DINO-AD methodology:
    1. K-means clustering on reference foreground features
    2. Reference self-similarity baseline (delta scoring)
    3. Direct patch-to-patch cosine distance
    4. Multiplicative combination (soft AND gate)
    5. Spatial consistency filtering + per-slice suppression

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] — new scan patch features
    features_ref : [D, Hp, Wp, C] — registered reference patch features
    volume_hu_new : [D_hu, H, W] — HU volume for foreground masking
    patch_size : ViT patch size (for mask computation)
    n_clusters : K for K-means (paper uses K=2)

    Returns
    -------
    anomaly_scores : [D, Hp, Wp] float32, in [0, 1]
    """
    D, Hp, Wp, C = features_new.shape
    eps = 1e-8

    # 1. Create foreground mask (tighter: 25% tissue, with erosion)
    fg_mask = None
    if volume_hu_new is not None:
        fg_mask = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg = fg_mask.sum()
        n_total = fg_mask.size
        print(f"  Foreground mask: {n_fg}/{n_total} patches ({100*n_fg/n_total:.1f}%)")

    # 2. L2-normalize all features
    ref_norm = features_ref / (np.linalg.norm(features_ref, axis=-1, keepdims=True) + eps)
    new_norm = features_new / (np.linalg.norm(features_new, axis=-1, keepdims=True) + eps)

    # 3. Collect reference foreground features for K-means
    if fg_mask is not None and fg_mask.any():
        ref_fg = ref_norm[fg_mask]
    else:
        ref_fg = ref_norm.reshape(-1, C)

    max_kmeans_samples = 50_000
    if ref_fg.shape[0] > max_kmeans_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(ref_fg.shape[0], max_kmeans_samples, replace=False)
        ref_fg_sample = ref_fg[indices]
    else:
        ref_fg_sample = ref_fg

    # 4. K-means clustering on reference foreground
    print(f"  Running K-means (K={n_clusters}) on {ref_fg_sample.shape[0]} reference patches...")
    centroids = _kmeans_simple(ref_fg_sample, k=n_clusters)
    centroids = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + eps)
    print(f"  Centroids shape: {centroids.shape}")

    # 5. Cosine similarity: new scan vs centroids
    query_flat = new_norm.reshape(-1, C)
    new_sim = (query_flat @ centroids.T).mean(axis=1).reshape(D, Hp, Wp)

    # 6. Reference self-similarity baseline
    ref_flat = ref_norm.reshape(-1, C)
    ref_sim = (ref_flat @ centroids.T).mean(axis=1).reshape(D, Hp, Wp)

    # Delta: how much the new scan dropped in similarity vs the reference baseline
    delta_sim = ref_sim - new_sim  # positive = new scan is less similar to centroids
    delta_sim = np.maximum(delta_sim, 0.0)  # only care about drops

    # 7. Direct patch-to-patch cosine distance
    direct_dist = 1.0 - np.sum(new_norm * ref_norm, axis=-1)  # [D, Hp, Wp], range [0, 2]
    direct_dist = np.maximum(direct_dist, 0.0)

    # Debug: print channel stats
    if fg_mask is not None and fg_mask.any():
        fg_delta = delta_sim[fg_mask]
        fg_direct = direct_dist[fg_mask]
        print(f"  Delta-sim (fg): min={fg_delta.min():.4f}, max={fg_delta.max():.4f}, "
              f"mean={fg_delta.mean():.4f}, p95={np.percentile(fg_delta, 95):.4f}")
        print(f"  Direct-dist (fg): min={fg_direct.min():.4f}, max={fg_direct.max():.4f}, "
              f"mean={fg_direct.mean():.4f}, p95={np.percentile(fg_direct, 95):.4f}")

    # 8. Percentile normalization (p2/p98) for both channels
    delta_norm = _percentile_normalize(delta_sim, fg_mask)
    direct_norm = _percentile_normalize(direct_dist, fg_mask)

    # 9. Multiplicative combination with safety floor
    #    A region must score high on BOTH channels to be flagged
    #    Floor at 0.05 to avoid complete suppression of extreme single-channel signals
    anomaly_scores = np.maximum(delta_norm, 0.05) * np.maximum(direct_norm, 0.05)

    # Re-normalize to [0, 1] after multiplication
    anomaly_scores = _percentile_normalize(anomaly_scores, fg_mask)

    # 10. Spatial consistency filter — suppress isolated patches
    anomaly_scores = _spatial_consistency_filter(anomaly_scores, fg_mask)

    # 11. Per-slice suppression — dampen globally-anomalous slices
    anomaly_scores = _per_slice_suppression(anomaly_scores, fg_mask)

    # 12. Light spatial smoothing (reduced sigma)
    anomaly_scores = gaussian_filter(
        anomaly_scores.astype(np.float64),
        sigma=[0.3, 0.5, 0.5],
    ).astype(np.float32)

    # 13. Zero out background
    if fg_mask is not None:
        anomaly_scores[~fg_mask] = 0.0

    # Debug stats
    if fg_mask is not None and fg_mask.any():
        fg_scores = anomaly_scores[fg_mask]
        print(f"  Final scores (fg): min={fg_scores.min():.3f}, "
              f"max={fg_scores.max():.3f}, mean={fg_scores.mean():.3f}, "
              f"p90={np.percentile(fg_scores, 90):.3f}, "
              f"p95={np.percentile(fg_scores, 95):.3f}, "
              f"p99={np.percentile(fg_scores, 99):.3f}")
    else:
        print(f"  Final scores: min={anomaly_scores.min():.3f}, "
              f"max={anomaly_scores.max():.3f}, mean={anomaly_scores.mean():.3f}")

    return anomaly_scores


def upsample_scores(
    scores: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Upsample scores to target resolution using bilinear interpolation."""
    factors = tuple(t / s for t, s in zip(target_shape, scores.shape))
    return zoom(scores, factors, order=1).astype(np.float32)
