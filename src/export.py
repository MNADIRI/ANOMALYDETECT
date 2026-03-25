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
    z_scores : [D, H, W] float32 at native resolution
    source_dicom_files : sorted list of .dcm paths for scan T
    threshold : z-score threshold for flagging
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

    # Create binary masks
    # Segment 1: "vigilance" zone – z > threshold
    mask_vigilance = z_scores > threshold
    # Segment 2: "alert" zone – z > threshold * 1.5
    mask_alert = z_scores > (threshold * 1.5)

    # Remove alert areas from vigilance (so they don't overlap)
    mask_vigilance_only = mask_vigilance & ~mask_alert

    has_vigilance = mask_vigilance_only.any()
    has_alert = mask_alert.any()

    if not has_vigilance and not has_alert:
        # Lower threshold to get at least some detections
        # Use a percentile-based approach
        p95 = np.percentile(z_scores, 95)
        if p95 > 1.0:
            mask_vigilance_only = z_scores > p95
            mask_alert = z_scores > np.percentile(z_scores, 99)
            mask_vigilance_only = mask_vigilance_only & ~mask_alert
            has_vigilance = mask_vigilance_only.any()
            has_alert = mask_alert.any()

    # Build segment descriptions
    segments = []
    masks = []

    if has_vigilance or not has_alert:
        # Always include at least one segment
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
            # Create a minimal mask (single voxel) to avoid empty segment
            mask_vigilance_only = z_scores > np.percentile(z_scores, 99.5)
            if not mask_vigilance_only.any():
                mask_vigilance_only[0, 0, 0] = True
        masks.append(mask_vigilance_only)

    if has_alert:
        seg_alert = hd.seg.SegmentDescription(
            segment_number=len(segments) + 1,
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

    # Stack masks: [n_segments, D, H, W] as boolean
    pixel_array = np.stack(masks, axis=0).astype(np.uint8)

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
        content_creator_name="CT Control Volume Pipeline",
    )

    seg.save_as(output_path)
    return output_path
