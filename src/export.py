"""
DICOM SEG export using highdicom.

Creates a DICOM Segmentation object that references the source CT images,
allowing Weasis to display the segmentation as an overlay.
"""

import datetime

import highdicom as hd
import numpy as np
import pydicom
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label


def _clean_mask(mask: np.ndarray, min_component_size: int = 20) -> np.ndarray:
    """Morphological cleanup: remove noise, fill holes, remove small components."""
    if not mask.any():
        return mask
    # Close small gaps (dilation then erosion)
    mask = binary_closing(mask, iterations=2)
    # Fill internal holes per slice (3D fill can leak across slices)
    for s in range(mask.shape[0]):
        mask[s] = binary_fill_holes(mask[s])
    # Remove small isolated 3D components
    labeled, n_comp = label(mask)
    for c in range(1, n_comp + 1):
        if (labeled == c).sum() < min_component_size:
            mask[labeled == c] = False
    # Open to smooth edges (erosion then dilation)
    mask = binary_opening(mask, iterations=1)
    return mask.astype(bool)


def create_dicom_seg(
    z_scores: np.ndarray,
    source_dicom_files: list[str],
    threshold: float = 3.0,
    output_path: str = "output.dcm",
) -> str:
    """
    Create a DICOM SEG file with change-detection segments.

    Parameters
    ----------
    z_scores : [D, H, W] float32 at native resolution (z-score values)
    source_dicom_files : sorted list of .dcm paths for scan T
    threshold : z-score threshold for vigilance (alert = threshold * 1.5)
    output_path : where to write the .dcm

    Returns
    -------
    output_path
    """
    # Read source DICOM images
    source_images = []
    for f in source_dicom_files:
        ds = pydicom.dcmread(f)
        source_images.append(ds)

    D_src = len(source_images)
    D_scores = z_scores.shape[0]

    # If dimensions don't match, resize z_scores to match source count
    if D_scores != D_src:
        from scipy.ndimage import zoom
        scale_z = D_src / D_scores
        scale_h = source_images[0].Rows / z_scores.shape[1]
        scale_w = source_images[0].Columns / z_scores.shape[2]
        z_scores = zoom(z_scores, (scale_z, scale_h, scale_w), order=1).astype(
            np.float32
        )

    # Ensure spatial dimensions match source
    target_h = source_images[0].Rows
    target_w = source_images[0].Columns
    if z_scores.shape[1] != target_h or z_scores.shape[2] != target_w:
        from scipy.ndimage import zoom
        scale_h = target_h / z_scores.shape[1]
        scale_w = target_w / z_scores.shape[2]
        z_scores = zoom(z_scores, (1.0, scale_h, scale_w), order=1).astype(
            np.float32
        )

    # Absolute z-score thresholding
    # Scores from scoring.py are already z-scores (global median + MAD normalized).
    # The user slider directly controls the z-score cutoff:
    #   threshold=1.5 → vigilance z>1.5 (sensitive)
    #   threshold=3.0 → vigilance z>3.0 (default)
    #   threshold=6.0 → vigilance z>6.0 (strict)
    fg_scores = z_scores[z_scores > 0]

    if fg_scores.size == 0:
        print("  No foreground scores — generating minimal SEG")
        vigilance_threshold = 999.0
        alert_threshold = 999.0
    else:
        vigilance_threshold = threshold
        alert_threshold = threshold + 2.0

        print(f"  Thresholds: vigilance=z>{vigilance_threshold:.1f}, "
              f"alert=z>{alert_threshold:.1f}  "
              f"(fg scores: p95={np.percentile(fg_scores, 95):.2f}, "
              f"p99={np.percentile(fg_scores, 99):.2f}, "
              f"max={fg_scores.max():.2f})")

    # Segment 1: "vigilance" zone
    mask_vigilance = z_scores > vigilance_threshold
    # Segment 2: "alert" zone
    mask_alert = z_scores > alert_threshold

    # Morphological cleanup — fill holes, remove noise, smooth edges
    mask_vigilance = _clean_mask(mask_vigilance)
    mask_alert = _clean_mask(mask_alert)

    # Remove alert areas from vigilance (no overlap)
    mask_vigilance_only = mask_vigilance & ~mask_alert

    has_vigilance = mask_vigilance_only.any()
    has_alert = mask_alert.any()

    print(f"  Vigilance voxels: {mask_vigilance_only.sum()}, "
          f"Alert voxels: {mask_alert.sum()}")

    if not has_vigilance and not has_alert:
        print("  No voxels above threshold — generating minimal SEG file")

    # Build segment descriptions
    segments = []
    masks = []

    # Always include at least the vigilance segment (highdicom requires >= 1)
    seg_vigilance = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="Zone de vigilance",
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Neoplasm,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name="CT Control Volume",
            version="1.0",
            family=codes.cid7162.ArtificialIntelligence,
        ),
        tracking_uid=generate_uid(),
        tracking_id="vigilance_zone",
    )
    segments.append(seg_vigilance)
    if not mask_vigilance_only.any():
        # Empty mask — no false positives
        mask_vigilance_only = np.zeros_like(z_scores, dtype=bool)
        mask_vigilance_only[0, 0, 0] = True  # minimal single voxel for validity
    masks.append(mask_vigilance_only)

    if has_alert:
        seg_alert = hd.seg.SegmentDescription(
            segment_number=2,
            segment_label="Zone d'alerte",
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="CT Control Volume",
                version="1.0",
                family=codes.cid7162.ArtificialIntelligence,
            ),
            tracking_uid=generate_uid(),
            tracking_id="alert_zone",
        )
        segments.append(seg_alert)
        masks.append(mask_alert)

    # highdicom 0.27 expects pixel_array shape [D, H, W, n_segments] for BINARY
    pixel_array = np.stack(masks, axis=-1).astype(np.bool_)

    # Create the DICOM SEG
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=pixel_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segments,
        series_instance_uid=generate_uid(),
        series_number=999,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="CT Control Volume",
        manufacturer_model_name="DINOv2 Anomaly Detection",
        software_versions="1.0",
        device_serial_number="0001",
        series_description="AI Control Volume - Change Detection",
        content_description="Automated change detection between CT scans",
        content_creator_name="CT Control Volume^Pipeline",
    )

    seg.save_as(output_path)
    return output_path
