"""
Anomaly scoring: PatchCore-style memory bank nearest neighbor.

Builds a gallery of all foreground patches from the reference (normal) scan.
For each new scan patch, finds the nearest neighbor in the gallery.
Anomaly score = cosine distance to nearest neighbor.
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
# Scoring: PatchCore-style memory bank nearest neighbor
# ---------------------------------------------------------------------------

MAX_BANK_SIZE = 60_000  # use all foreground patches (no subsampling)
CHUNK_SIZE = 500        # process new patches in chunks to limit memory
EDGE_SLICES = 10        # trim first/last N slices (different scan coverage)


def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    volume_hu_ref: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute anomaly scores using memory bank nearest neighbor (PatchCore).

    Builds a gallery of all foreground reference patches. For each new
    patch, finds the nearest neighbor in the gallery via cosine similarity.
    Anomaly score = cosine distance to nearest neighbor.

    This works for cross-patient comparison: normal tissues (GM, WM, CSF)
    find good matches across patients; pathology (hemorrhage) doesn't.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] — new scan features (L2-normalized)
    features_ref : [D_ref, Hp, Wp, C] — reference features (L2-normalized)
    volume_hu_new : [D_hu, H, W] — HU volume for new scan foreground mask
    volume_hu_ref : [D_hu_ref, H, W] — HU volume for reference foreground mask
    patch_size : ViT patch size (unused, kept for API compat)

    Returns
    -------
    scores : [D, Hp, Wp] float32 — cosine distance to nearest reference patch
    """
    D, Hp, Wp, C = features_new.shape
    D_ref = features_ref.shape[0]
    eps = 1e-8

    # 1. Build foreground masks
    fg_mask_new = None
    if volume_hu_new is not None:
        fg_mask_new = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg_new = fg_mask_new.sum()
        print(f"  Foreground mask (new): {n_fg_new}/{fg_mask_new.size} patches "
              f"({100 * n_fg_new / fg_mask_new.size:.1f}%)")

    fg_mask_ref = None
    if volume_hu_ref is not None:
        fg_mask_ref = _create_foreground_mask(volume_hu_ref, (D_ref, Hp, Wp))
        n_fg_ref = fg_mask_ref.sum()
        print(f"  Foreground mask (ref): {n_fg_ref}/{fg_mask_ref.size} patches "
              f"({100 * n_fg_ref / fg_mask_ref.size:.1f}%)")

    # 2. Build reference memory bank (foreground patches only)
    if fg_mask_ref is not None and fg_mask_ref.any():
        ref_bank = features_ref[fg_mask_ref]  # [N_fg_ref, C]
    else:
        ref_bank = features_ref.reshape(-1, C)

    # Subsample for speed if too large
    if ref_bank.shape[0] > MAX_BANK_SIZE:
        rng = np.random.default_rng(42)
        idx = rng.choice(ref_bank.shape[0], MAX_BANK_SIZE, replace=False)
        ref_bank = ref_bank[idx]

    # L2-normalize the bank (should already be, but safety)
    ref_bank = ref_bank / (np.linalg.norm(ref_bank, axis=-1, keepdims=True) + eps)
    print(f"  Memory bank: {ref_bank.shape[0]} reference patches, dim={C}")

    # 3. For each new patch, find nearest neighbor in bank
    new_flat = features_new.reshape(-1, C)
    new_flat = new_flat / (np.linalg.norm(new_flat, axis=-1, keepdims=True) + eps)

    n_new = new_flat.shape[0]
    nn_distances = np.zeros(n_new, dtype=np.float32)

    # Process in chunks to limit memory: chunk × bank matrix
    for start in range(0, n_new, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_new)
        chunk = new_flat[start:end]  # [chunk_size, C]

        # Cosine similarity: [chunk_size, bank_size]
        sim = chunk @ ref_bank.T
        best_sim = sim.max(axis=1)  # [chunk_size]
        nn_distances[start:end] = 1.0 - best_sim

    nn_distances = np.maximum(nn_distances, 0.0)
    distances = nn_distances.reshape(D, Hp, Wp)

    # 3b. Per-slice z-score normalization (removes edge-to-center gradient)
    #     Each slice gets its own baseline so hemorrhage stands out locally
    for s in range(D):
        if fg_mask_new is not None and fg_mask_new[s].any():
            fg_vals = distances[s][fg_mask_new[s]]
            median_s = np.median(fg_vals)
            mad_s = np.median(np.abs(fg_vals - median_s))
            mad_s = max(mad_s, 1e-6)
            distances[s] = np.maximum(
                (distances[s] - median_s) / (1.4826 * mad_s), 0.0
            )
        else:
            distances[s] = 0.0

    # 4. Light Gaussian smoothing
    distances = gaussian_filter(
        distances.astype(np.float64),
        sigma=[0.3, 0.7, 0.7],
    ).astype(np.float32)

    # 5. Mask background and edge slices
    if fg_mask_new is not None:
        distances[~fg_mask_new] = 0.0
    if D > 2 * EDGE_SLICES:
        distances[:EDGE_SLICES] = 0.0
        distances[-EDGE_SLICES:] = 0.0
        print(f"  Edge trimming: zeroed slices 0-{EDGE_SLICES-1} and {D-EDGE_SLICES}-{D-1}")

    # 6. Debug stats
    if fg_mask_new is not None and fg_mask_new.any():
        fg_d = distances[fg_mask_new]
        print(f"  NN distances (fg): min={fg_d.min():.4f}, max={fg_d.max():.4f}, "
              f"mean={fg_d.mean():.4f}, "
              f"p95={np.percentile(fg_d, 95):.4f}, "
              f"p99={np.percentile(fg_d, 99):.4f}")

        # Per-slice analysis
        slice_means = []
        for s in range(D):
            if fg_mask_new is not None and fg_mask_new[s].any():
                slice_means.append(distances[s][fg_mask_new[s]].mean())
            else:
                slice_means.append(0.0)
        slice_means = np.array(slice_means)
        top5 = np.argsort(slice_means)[-5:][::-1]
        print(f"  Top 5 slices by mean NN distance: {top5}")
        for s in top5:
            print(f"    Slice {s}: mean={slice_means[s]:.4f}, "
                  f"max={distances[s].max():.4f}")

        # Full per-slice profile (every 10th slice)
        print(f"  Per-slice distance profile (every 10th):")
        for s in range(0, D, 10):
            sm = slice_means[s]
            bar = "#" * int(sm * 100)
            print(f"    Slice {s:3d}: mean={sm:.4f} {bar}")

        np.save("outputs/diagnostic_slice_profile.npy", slice_means)
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
