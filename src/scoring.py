"""
Anomaly scoring using DINO-AD methodology:
- Foreground-aware K-means clustering (K=2) on reference features
- Cosine similarity to cluster centroids
- Min-max normalization
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_closing


def _create_foreground_mask(
    volume_hu: np.ndarray,
    grid_shape: tuple[int, int, int],
    hu_threshold: float = -900.0,
    min_tissue_fraction: float = 0.1,
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

    # Morphological closing to fill small holes (per-slice)
    for d in range(D_feat):
        mask[d] = binary_closing(mask[d], iterations=2)

    return mask


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
        # Distance to nearest existing centroid
        dists = np.min(
            [np.sum((features - centroids[j]) ** 2, axis=1) for j in range(i)],
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-10)
        centroids[i] = features[rng.choice(N, p=probs)]

    # Iterate
    for _ in range(max_iter):
        # Assign to nearest centroid
        dists = np.stack([
            np.sum((features - centroids[j]) ** 2, axis=1)
            for j in range(k)
        ], axis=1)  # [N, K]
        assignments = np.argmin(dists, axis=1)  # [N]

        # Update centroids
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
    Compute anomaly scores using DINO-AD methodology:
    1. K-means clustering on reference foreground features
    2. Cosine similarity between query patches and cluster centroids
    3. Min-max normalization

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

    # 1. Create foreground mask
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
        ref_fg = ref_norm[fg_mask]  # [N_fg, C]
    else:
        ref_fg = ref_norm.reshape(-1, C)

    # Subsample if too many (for speed)
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
    # L2-normalize centroids
    centroids = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + eps)
    print(f"  Centroids shape: {centroids.shape}")

    # 5. Compute cosine similarity between each query patch and all centroids
    #    sim[d, hp, wp, k] = dot(query[d,hp,wp], centroid[k])
    query_flat = new_norm.reshape(-1, C)  # [D*Hp*Wp, C]
    sim_matrix = query_flat @ centroids.T  # [D*Hp*Wp, K]
    mean_sim = sim_matrix.mean(axis=1)  # [D*Hp*Wp]
    mean_sim = mean_sim.reshape(D, Hp, Wp)

    # 6. Min-max normalize the similarity map
    if fg_mask is not None and fg_mask.any():
        fg_sims = mean_sim[fg_mask]
        sim_min = fg_sims.min()
        sim_max = fg_sims.max()
    else:
        sim_min = mean_sim.min()
        sim_max = mean_sim.max()

    sim_range = sim_max - sim_min
    if sim_range < eps:
        sim_range = 1.0

    normalized_sim = (mean_sim - sim_min) / sim_range
    normalized_sim = np.clip(normalized_sim, 0.0, 1.0)

    # 7. Anomaly = 1 - normalized similarity (lower similarity = higher anomaly)
    anomaly_scores = 1.0 - normalized_sim

    # 8. Light spatial smoothing (preserve focal lesions)
    anomaly_scores = gaussian_filter(
        anomaly_scores.astype(np.float64),
        sigma=[0.5, 0.8, 0.8],
    ).astype(np.float32)

    # 9. Zero out background
    if fg_mask is not None:
        anomaly_scores[~fg_mask] = 0.0

    # Debug stats
    if fg_mask is not None and fg_mask.any():
        fg_scores = anomaly_scores[fg_mask]
        print(f"  Anomaly scores (foreground): min={fg_scores.min():.3f}, "
              f"max={fg_scores.max():.3f}, mean={fg_scores.mean():.3f}, "
              f"p90={np.percentile(fg_scores, 90):.3f}, "
              f"p95={np.percentile(fg_scores, 95):.3f}, "
              f"p99={np.percentile(fg_scores, 99):.3f}")
    else:
        print(f"  Anomaly scores: min={anomaly_scores.min():.3f}, "
              f"max={anomaly_scores.max():.3f}, mean={anomaly_scores.mean():.3f}")

    return anomaly_scores


def upsample_scores(
    scores: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Upsample scores to target resolution using bilinear interpolation."""
    factors = tuple(t / s for t, s in zip(target_shape, scores.shape))
    return zoom(scores, factors, order=1).astype(np.float32)
