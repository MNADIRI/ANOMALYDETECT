"""
DICOM ingestion: reads a folder of DICOM CT files and produces
standardised HU volumes and adaptive 3-channel windowed volumes.
"""

import os
from collections import Counter

import numpy as np
import pydicom
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Region-aware HU windowing
# ---------------------------------------------------------------------------

REGION_PRESETS = {
    "brain": [
        ("brain",    40,   80),    # parenchyme vs hémorragie — Δ max
        ("subdural", 60,  200),    # vision élargie tissus mous + sang
        ("bone",    500, 2500),    # crâne, calcifications
    ],
    "chest": [
        ("mediastinum", 40,  400),
        ("lung",      -600, 1500),
        ("bone",       300, 1500),
    ],
    "abdomen": [
        ("soft",  40,  400),
        ("lung", -600, 1500),
        ("bone",  300, 1500),
    ],
}
REGION_PRESETS["default"] = REGION_PRESETS["abdomen"]


def detect_body_region(volume_hu: np.ndarray, metadata: dict) -> str:
    """
    Detect body region from DICOM tags or HU histogram analysis.

    Priority:
      1. DICOM BodyPartExamined / StudyDescription tags
      2. HU histogram heuristic (brain vs chest vs abdomen)
    """
    # --- Try DICOM tags first ---
    body_part = metadata.get("body_part_examined", "").upper()
    study_desc = metadata.get("study_description", "").upper()

    tag_text = f"{body_part} {study_desc}"
    brain_keywords = ["HEAD", "BRAIN", "CRANE", "CEREBR", "TETE", "CRÂNE"]
    chest_keywords = ["CHEST", "THORAX", "LUNG", "PULMON", "THORAC"]
    abdomen_keywords = ["ABDOMEN", "ABDOM", "PELVI"]

    for kw in brain_keywords:
        if kw in tag_text:
            return "brain"
    for kw in chest_keywords:
        if kw in tag_text:
            return "chest"
    for kw in abdomen_keywords:
        if kw in tag_text:
            return "abdomen"

    # --- HU histogram heuristic ---
    # Sample middle slices for speed
    D = volume_hu.shape[0]
    mid = D // 2
    sample = volume_hu[max(0, mid - 5):mid + 5]
    flat = sample.ravel()
    # Only consider tissue range
    tissue = flat[(flat > -100) & (flat < 200)]

    if tissue.size == 0:
        return "default"

    # Brain: tight distribution centred around 20-45 HU, no lung peak
    std_tissue = np.std(tissue)
    mean_tissue = np.mean(tissue)

    # Check for lung air peak (strong indicator of chest)
    air_voxels = flat[(flat > -900) & (flat < -400)]
    air_fraction = air_voxels.size / max(flat.size, 1)

    if air_fraction > 0.15:
        return "chest"
    if std_tissue < 40 and 10 < mean_tissue < 60:
        return "brain"

    return "default"


def _apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply a single HU window and normalise to [0, 1]."""
    lower = center - width / 2.0
    upper = center + width / 2.0
    out = np.clip(hu, lower, upper)
    out = (out - lower) / (upper - lower)
    return out.astype(np.float32)


def adaptive_triple_channel(
    hu_volume: np.ndarray,
    metadata: dict,
    body_region: str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Convert HU volume [D, H, W] → 3-channel [D, 3, H, W].

    Uses region-specific HU windows normalised to [0, 1].
    Returns (volume_3ch, detected_region).
    """
    if body_region is None:
        body_region = detect_body_region(hu_volume, metadata)

    preset = REGION_PRESETS.get(body_region, REGION_PRESETS["default"])
    print(f"  Body region: {body_region} → windows: "
          f"{[(n, c, w) for n, c, w in preset]}")

    channels = []
    for name, center, width in preset:
        windowed = _apply_window(hu_volume, center, width)
        channels.append(windowed)
        print(f"    Channel '{name}': range [{windowed.min():.3f}, {windowed.max():.3f}]")

    volume_3ch = np.stack(channels, axis=1)  # [D, 3, H, W]
    return volume_3ch, body_region


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

TARGET_SIZE = 518  # 37 × 14 — DINOv2 ViT-B14 native (no padding needed)


def ingest_dicom_folder(
    dicom_dir: str,
    body_region: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a DICOM folder and produce a standardised volume.

    Returns
    -------
    volume_3ch : np.ndarray float32 [D, 3, H, W]
        Triple-windowed volume (H = W = 512).
    volume_hu : np.ndarray float32 [D, H, W]
        HU mono-channel volume (resampled to same grid).
    metadata : dict
        spacing, origin, direction, patient_id, study_date,
        series_uid, source_files, original_shape.
    """
    # 1. Collect all DICOM files -------------------------------------------
    dcm_paths = []
    skip_names = {"dicomdir", "dicomdir."}
    for root, _dirs, files in os.walk(dicom_dir):
        for fname in files:
            # Skip DICOMDIR index files and hidden files
            if fname.lower().rstrip(".") in skip_names:
                continue
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(".dcm") or "." not in fname:
                dcm_paths.append(fpath)
            else:
                # Try to read any file — DICOM files don't always have .dcm extension
                try:
                    pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                    dcm_paths.append(fpath)
                except Exception:
                    pass

    if not dcm_paths:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # 2. Read headers (no pixels) ------------------------------------------
    headers = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            # Skip files without pixel data indicators (DICOMDIR, SR, etc.)
            has_rows = hasattr(ds, "Rows")
            has_cols = hasattr(ds, "Columns")
            if not (has_rows and has_cols):
                continue
            headers.append((p, ds))
        except Exception:
            continue

    if not headers:
        raise ValueError(
            f"No readable DICOM image files in {dicom_dir}. "
            f"Found {len(dcm_paths)} files but none contain image data. "
            f"Make sure to provide the folder with the actual CT slices, "
            f"not just the DICOMDIR file."
        )

    # 3. Filter CT only, pick largest series -------------------------------
    ct_headers = [
        (p, ds) for p, ds in headers
        if getattr(ds, "Modality", "").upper() == "CT"
    ]
    if not ct_headers:
        # Fallback: use all files with image data if no CT Modality tag
        ct_headers = headers

    series_counter = Counter(
        getattr(ds, "SeriesInstanceUID", "unknown") for _, ds in ct_headers
    )
    main_series = series_counter.most_common(1)[0][0]
    ct_headers = [
        (p, ds) for p, ds in ct_headers
        if getattr(ds, "SeriesInstanceUID", "unknown") == main_series
    ]

    # 4. Sort by ImagePositionPatient[2] -----------------------------------
    def _z_pos(item):
        ds = item[1]
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        return 0.0

    ct_headers.sort(key=_z_pos)
    sorted_paths = [p for p, _ in ct_headers]

    # 5. Read pixel data and convert to HU ---------------------------------
    slices_hu = []
    for p, ds in ct_headers:
        ds_full = pydicom.dcmread(p)
        arr = ds_full.pixel_array.astype(np.float64)
        slope = float(getattr(ds_full, "RescaleSlope", 1.0))
        intercept = float(getattr(ds_full, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        slices_hu.append(hu.astype(np.float32))

    volume_hu_raw = np.stack(slices_hu, axis=0)  # [D, H_orig, W_orig]

    # 6. Extract spatial metadata ------------------------------------------
    ds0 = ct_headers[0][1]
    pixel_spacing = [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    # Estimate slice spacing
    if len(ct_headers) >= 2:
        z0 = _z_pos(ct_headers[0])
        z1 = _z_pos(ct_headers[1])
        slice_spacing = abs(z1 - z0)
        if slice_spacing == 0:
            slice_spacing = float(getattr(ds0, "SliceThickness", 1.0))
    else:
        slice_spacing = float(getattr(ds0, "SliceThickness", 1.0))

    spacing_orig = (slice_spacing, pixel_spacing[0], pixel_spacing[1])  # (sz, sy, sx)
    origin_vals = [float(x) for x in getattr(ds0, "ImagePositionPatient", [0, 0, 0])]
    origin = (origin_vals[2], origin_vals[1], origin_vals[0])  # (oz, oy, ox) in z,y,x

    iop = [float(x) for x in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    direction = (
        iop[0], iop[1], iop[2],
        iop[3], iop[4], iop[5],
        0.0, 0.0, 1.0,
    )

    original_shape = volume_hu_raw.shape  # (D, H_orig, W_orig)

    # 7. Resample to isotropic in-plane at TARGET_SIZE ---------------------
    sitk_image = sitk.GetImageFromArray(volume_hu_raw)
    sitk_image.SetSpacing((spacing_orig[2], spacing_orig[1], spacing_orig[0]))  # (sx, sy, sz)
    sitk_image.SetOrigin((origin_vals[0], origin_vals[1], origin_vals[2]))

    orig_size = sitk_image.GetSize()  # (W, H, D) in SimpleITK order
    orig_spacing = sitk_image.GetSpacing()

    # Compute new spacing for TARGET_SIZE in-plane
    # Keep the larger of the two in-plane dimensions to set the FOV
    fov_x = orig_size[0] * orig_spacing[0]
    fov_y = orig_size[1] * orig_spacing[1]
    fov_max = max(fov_x, fov_y)
    new_in_plane_spacing = fov_max / TARGET_SIZE

    new_spacing = (new_in_plane_spacing, new_in_plane_spacing, orig_spacing[2])
    new_size = (TARGET_SIZE, TARGET_SIZE, orig_size[2])

    # Center the resampled volume
    center_phys = [
        sitk_image.GetOrigin()[i] + orig_size[i] * orig_spacing[i] / 2.0
        for i in range(3)
    ]
    new_origin = [
        center_phys[i] - new_size[i] * new_spacing[i] / 2.0
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1024.0)

    resampled = resampler.Execute(sitk_image)
    volume_hu = sitk.GetArrayFromImage(resampled).astype(np.float32)  # [D, H, W]

    # 8. Metadata (build early so adaptive windowing can read DICOM tags) ----
    metadata = {
        "spacing": (
            new_spacing[2],  # sz
            new_spacing[1],  # sy
            new_spacing[0],  # sx
        ),
        "origin": tuple(resampled.GetOrigin()),
        "direction": resampled.GetDirection(),
        "patient_id": str(getattr(ds0, "PatientID", "UNKNOWN")),
        "study_date": str(getattr(ds0, "StudyDate", "")),
        "series_uid": str(getattr(ds0, "SeriesInstanceUID", "")),
        "body_part_examined": str(getattr(ds0, "BodyPartExamined", "")),
        "study_description": str(getattr(ds0, "StudyDescription", "")),
        "source_files": sorted_paths,
        "original_shape": original_shape,
        "sitk_reference": resampled,  # keep for registration
        "original_spacing": spacing_orig,
        "original_origin": origin_vals,
        "original_direction": sitk_image.GetDirection(),
    }

    # 9. Adaptive CLAHE windowing -------------------------------------------
    volume_3ch, detected_region = adaptive_triple_channel(
        volume_hu, metadata, body_region=body_region,
    )
    metadata["body_region"] = detected_region

    return volume_3ch, volume_hu, metadata
