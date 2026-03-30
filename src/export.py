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

    # Percentile-based thresholding on foreground scores
    fg_scores = z_scores[z_scores > 0]

    if fg_scores.size == 0:
        print("  No foreground scores — generating minimal SEG")
        vigilance_threshold = 999.0
        alert_threshold = 999.0
    else:
        # Map user slider (1.5–6.0) to percentiles
        # threshold=3.0 (default) → vigilance at p95, alert at p96.5
        # threshold=1.5 (sensitive) → vigilance at p92.5, alert at p94
        # threshold=6.0 (strict) → vigilance at p100 (capped at p99.9)
        pct_vigilance = min(90 + threshold * (10.0 / 6.0), 99.9)
        pct_alert = min(pct_vigilance + 1.5, 99.95)

        vigilance_threshold = np.percentile(fg_scores, pct_vigilance)
        alert_threshold = np.percentile(fg_scores, pct_alert)

        print(f"  Thresholds: vigilance={vigilance_threshold:.6f} "
              f"(p{pct_vigilance:.1f}), "
              f"alert={alert_threshold:.6f} (p{pct_alert:.1f})")

    # Segment 1: "vigilance" zone
    mask_vigilance = z_scores > vigilance_threshold
    # Segment 2: "alert" zone
    mask_alert = z_scores > alert_threshold

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
