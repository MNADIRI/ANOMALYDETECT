"""
Anomaly scoring: Position-aware local nearest neighbor + HU validation.

Exploits spatial correspondence from registration + slice matching.
For each patch (s, i, j) in the new scan, compares only to patches
in a local spatial neighborhood around (s, i, j) in the registered
reference. HU difference amplifies feature anomalies where density
actually changed (hemorrhage = HU increase).
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
# Position-aware local scoring
# ---------------------------------------------------------------------------

SPATIAL_WINDOW = 3   # ±3 patches in i, j
SLICE_WINDOW = 1     # ±1 slice in z
EDGE_SLICES = 10     # trim first/last N slices


def _position_aware_distance(
    feat_new: np.ndarray,
    feat_ref: np.ndarray,
) -> np.ndarray:
    """
    Position-aware local nearest neighbor cosine distance.

    For each patch (s, i, j) in feat_new, finds the best cosine match
    in feat_ref within a local neighborhood [s±SLICE_WINDOW, i±SPATIAL_WINDOW,
    j±SPATIAL_WINDOW].

    Processes slice-by-slice to keep memory usage low (~12 MB temporaries
    instead of ~2 GB when operating on the full volume).

    Parameters
    ----------
    feat_new : [D, Hp, Wp, C] — new scan features
    feat_ref : [D, Hp, Wp, C] — registered+slice-matched reference features

    Returns
    -------
    distances : [D, Hp, Wp] — cosine distance to best local match
    """
    D, Hp, Wp, C = feat_new.shape
    eps = 1e-8

    best_sim = np.full((D, Hp, Wp), -1.0, dtype=np.float32)

    # Cache for normalized ref slices (avoid redundant computation)
    ref_norm_cache: dict[int, np.ndarray] = {}

    def _get_ref_normalized(s_ref: int) -> np.ndarray:
        if s_ref not in ref_norm_cache:
            fr = feat_ref[s_ref].astype(np.float32)
            norms = np.linalg.norm(fr, axis=-1, keepdims=True)
            ref_norm_cache[s_ref] = fr / (norms + eps)
            # Keep cache small: only need current ± SLICE_WINDOW
            stale = [k for k in ref_norm_cache if k < s_ref - SLICE_WINDOW - 1]
            for k in stale:
                del ref_norm_cache[k]
        return ref_norm_cache[s_ref]

    # Pre-compute spatial offset ranges (same for every slice)
    offsets_ij = []
    for di in range(-SPATIAL_WINDOW, SPATIAL_WINDOW + 1):
        for dj in range(-SPATIAL_WINDOW, SPATIAL_WINDOW + 1):
            i_new_lo = max(0, -di)
            i_new_hi = min(Hp, Hp - di)
            i_ref_lo = max(0, di)
            i_ref_hi = min(Hp, Hp + di)
            j_new_lo = max(0, -dj)
            j_new_hi = min(Wp, Wp - dj)
            j_ref_lo = max(0, dj)
            j_ref_hi = min(Wp, Wp + dj)
            if i_new_hi > i_new_lo and j_new_hi > j_new_lo:
                offsets_ij.append((
                    i_new_lo, i_new_hi, i_ref_lo, i_ref_hi,
                    j_new_lo, j_new_hi, j_ref_lo, j_ref_hi,
                ))

    n_offsets = (2 * SLICE_WINDOW + 1) * len(offsets_ij)

    for s in range(D):
        # Normalize new slice (~12 MB, not 2.1 GB)
        fn = feat_new[s].astype(np.float32)
        fn_norms = np.linalg.norm(fn, axis=-1, keepdims=True)
        fn = fn / (fn_norms + eps)

        for ds in range(-SLICE_WINDOW, SLICE_WINDOW + 1):
            s_ref = s + ds
            if s_ref < 0 or s_ref >= D:
                continue
            fr = _get_ref_normalized(s_ref)

            for (inl, inh, irl, irh, jnl, jnh, jrl, jrh) in offsets_ij:
                sim = (fn[inl:inh, jnl:jnh, :] * fr[irl:irh, jrl:jrh, :]).sum(axis=-1)
                best_sim[s, inl:inh, jnl:jnh] = np.maximum(
                    best_sim[s, inl:inh, jnl:jnh], sim,
                )

        if s % 25 == 0:
            print(f"    Position-aware: slice {s}/{D}...")

    print(f"  Position-aware scoring: {n_offsets} offsets/slice "
          f"(z=±{SLICE_WINDOW}, xy=±{SPATIAL_WINDOW}), {D} slices")

    return 1.0 - np.maximum(best_sim, 0.0)


# ---------------------------------------------------------------------------
# HU difference scoring (patch-level)
# ---------------------------------------------------------------------------

def _compute_hu_diff(
    vol_hu_new: np.ndarray,
    vol_hu_ref: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Compute mean HU difference per patch block: new - ref.

    Positive values = density increase (potential hemorrhage).
    """
    D, Hp, Wp = grid_shape
    D_hu, H_hu, W_hu = vol_hu_new.shape

    slice_ratio = D_hu / D
    patch_h = H_hu / Hp
    patch_w = W_hu / Wp

    hu_diff = np.zeros((D, Hp, Wp), dtype=np.float32)

    for d in range(D):
        d_s = int(d * slice_ratio)
        d_e = max(d_s + 1, min(int((d + 1) * slice_ratio), D_hu))
        for i in range(Hp):
            h_s = int(i * patch_h)
            h_e = max(h_s + 1, min(int((i + 1) * patch_h), H_hu))
            for j in range(Wp):
                w_s = int(j * patch_w)
                w_e = max(w_s + 1, min(int((j + 1) * patch_w), W_hu))
                block_new = vol_hu_new[d_s:d_e, h_s:h_e, w_s:w_e]
                block_ref = vol_hu_ref[d_s:d_e, h_s:h_e, w_s:w_e]
                hu_diff[d, i, j] = block_new.mean() - block_ref.mean()

    return hu_diff


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_change_scores(
    features_new: np.ndarray,
    features_ref: np.ndarray,
    volume_hu_new: np.ndarray | None = None,
    volume_hu_ref: np.ndarray | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Compute anomaly scores using position-aware local NN + HU validation.

    Exploits spatial correspondence from registration + slice matching:
    feat_ref[s, i, j] corresponds to the same anatomical position as
    feat_new[s, i, j]. Compares each new patch only to a local spatial
    neighborhood in the reference, then amplifies with HU difference.

    Mathematical definition of "pathological":
    A patch is pathological if its local cosine distance is a statistical
    outlier (global z-score > threshold) AND/OR HU density changed.

    Parameters
    ----------
    features_new : [D, Hp, Wp, C] — new scan features (L2-normalized)
    features_ref : [D, Hp, Wp, C] — reference features (slice-matched, same D)
    volume_hu_new : [D_hu, H, W] — HU volume for new scan
    volume_hu_ref : [D_hu, H, W] — HU volume for reference (registered)
    patch_size : ViT patch size (unused, kept for API compat)

    Returns
    -------
    scores : [D, Hp, Wp] float32 — anomaly z-scores
    """
    D, Hp, Wp, C = features_new.shape
    eps = 1e-8

    # 1. Build foreground masks
    fg_mask_new = None
    if volume_hu_new is not None:
        fg_mask_new = _create_foreground_mask(volume_hu_new, (D, Hp, Wp))
        n_fg_new = fg_mask_new.sum()
        print(f"  Foreground mask (new): {n_fg_new}/{fg_mask_new.size} patches "
              f"({100 * n_fg_new / fg_mask_new.size:.1f}%)")

    if volume_hu_ref is not None:
        fg_mask_ref = _create_foreground_mask(
            volume_hu_ref, (features_ref.shape[0], Hp, Wp),
        )
        n_fg_ref = fg_mask_ref.sum()
        print(f"  Foreground mask (ref): {n_fg_ref}/{fg_mask_ref.size} patches "
              f"({100 * n_fg_ref / fg_mask_ref.size:.1f}%)")

    # 2. Position-aware local nearest neighbor distance
    distances = _position_aware_distance(features_new, features_ref)

    # 3. HU difference boost (amplifies where density actually changed)
    if volume_hu_new is not None and volume_hu_ref is not None:
        hu_diff = _compute_hu_diff(volume_hu_new, volume_hu_ref, (D, Hp, Wp))

        # Hemorrhage = density increase: boost score where HU went up
        # +50 HU → boost factor 1.0 (doubles the score)
        # +10 HU → boost factor 0.0 (no effect)
        # negative → no effect (density decrease = not hemorrhage)
        hu_boost = np.clip(np.maximum(hu_diff - 10.0, 0.0) / 40.0, 0.0, 2.0)
        distances = distances * (1.0 + hu_boost)

        # Diagnostic
        if fg_mask_new is not None and fg_mask_new.any():
            hu_fg = hu_diff[fg_mask_new]
            print(f"  HU diff (fg): min={hu_fg.min():.1f}, max={hu_fg.max():.1f}, "
                  f"mean={hu_fg.mean():.1f}, "
                  f"patches with ΔHU>20: {(hu_fg > 20).sum()}")

    # 4. Global z-score normalization
    #    Position-aware scoring produces low baseline for normal tissue,
    #    so global normalization correctly identifies outliers.
    if fg_mask_new is not None and fg_mask_new.any():
        fg_vals = distances[fg_mask_new]
        global_median = np.median(fg_vals)
        global_mad = np.median(np.abs(fg_vals - global_median))
        global_mad = max(global_mad, eps)
        print(f"  Global normalization: median={global_median:.4f}, "
              f"MAD={global_mad:.4f}, σ_est={1.4826 * global_mad:.4f}")
        distances = np.maximum(
            (distances - global_median) / (1.4826 * global_mad), 0.0
        )
    else:
        print("  WARNING: no foreground patches detected")

    # 5. Gaussian smoothing
    distances = gaussian_filter(
        distances.astype(np.float64),
        sigma=[0.5, 1.0, 1.0],
    ).astype(np.float32)

    # 6. Mask background and edge slices
    if fg_mask_new is not None:
        distances[~fg_mask_new] = 0.0
    if D > 2 * EDGE_SLICES:
        distances[:EDGE_SLICES] = 0.0
        distances[-EDGE_SLICES:] = 0.0
        print(f"  Edge trimming: zeroed slices 0-{EDGE_SLICES - 1} "
              f"and {D - EDGE_SLICES}-{D - 1}")

    # 7. Debug stats
    if fg_mask_new is not None and fg_mask_new.any():
        fg_d = distances[fg_mask_new]
        fg_nonzero = fg_d[fg_d > 0]
        print(f"  Final scores (fg): min={fg_d.min():.4f}, max={fg_d.max():.4f}, "
              f"mean={fg_d.mean():.4f}, "
              f"p95={np.percentile(fg_d, 95):.4f}, "
              f"p99={np.percentile(fg_d, 99):.4f}")
        print(f"  Non-zero fg patches: {fg_nonzero.size}/{fg_d.size} "
              f"({100 * fg_nonzero.size / max(fg_d.size, 1):.1f}%)")

        # Per-slice analysis
        slice_means = []
        for s in range(D):
            if fg_mask_new is not None and fg_mask_new[s].any():
                slice_means.append(distances[s][fg_mask_new[s]].mean())
            else:
                slice_means.append(0.0)
        slice_means = np.array(slice_means)
        top5 = np.argsort(slice_means)[-5:][::-1]
        print(f"  Top 5 slices by mean score: {top5}")
        for s in top5:
            print(f"    Slice {s}: mean={slice_means[s]:.4f}, "
                  f"max={distances[s].max():.4f}")

        # Per-slice profile (every 10th)
        print(f"  Per-slice score profile (every 10th):")
        for s in range(0, D, 10):
            sm = slice_means[s]
            bar = "#" * int(sm * 20)
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
