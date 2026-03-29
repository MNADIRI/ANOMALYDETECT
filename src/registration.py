"""
Rigid registration of the reference volume (T-1) onto the new volume (T)
using SimpleITK Euler3D + Mattes Mutual Information.
"""

import numpy as np
import SimpleITK as sitk


def _metadata_matches(meta_a: dict, meta_b: dict) -> bool:
    """Check if two scans are the same acquisition (same SeriesInstanceUID)."""
    uid_a = meta_a.get("series_uid")
    uid_b = meta_b.get("series_uid")
    return bool(uid_a and uid_b and uid_a == uid_b)


def _match_dimensions(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop or pad a volume to match target shape (no resampling)."""
    result = np.full(target_shape, -1024.0, dtype=np.float32)
    # Copy overlapping region
    d = min(volume.shape[0], target_shape[0])
    h = min(volume.shape[1], target_shape[1])
    w = min(volume.shape[2], target_shape[2])
    result[:d, :h, :w] = volume[:d, :h, :w]
    return result


def _numpy_to_sitk(
    volume: np.ndarray,
    spacing: tuple,
    origin: tuple,
    direction: tuple,
) -> sitk.Image:
    """Convert a numpy [D, H, W] array to a SimpleITK Image with metadata."""
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    # SimpleITK expects (sx, sy, sz) ordering
    img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    img.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    if len(direction) == 9:
        img.SetDirection(tuple(float(d) for d in direction))
    return img


def register_to_reference(
    fixed_hu: np.ndarray,
    moving_hu: np.ndarray,
    fixed_meta: dict,
    moving_meta: dict,
) -> tuple[np.ndarray, sitk.Transform, sitk.Image]:
    """
    Rigidly register the moving volume (T-1) onto the fixed volume (T).

    Parameters
    ----------
    fixed_hu : [D, H, W] float32 – new scan T (target space)
    moving_hu : [D, H, W] float32 – reference scan T-1
    fixed_meta, moving_meta : dicts from ingestion with spacing/origin/direction

    Returns
    -------
    registered_hu : [D, H, W] float32 – T-1 aligned to T
    transform : sitk.Transform – the computed rigid transform
    fixed_image : sitk.Image – reference image for resampling other volumes
    """
    fixed_image = fixed_meta.get("sitk_reference")
    if fixed_image is None:
        fixed_image = _numpy_to_sitk(
            fixed_hu, fixed_meta["spacing"],
            fixed_meta["original_origin"], fixed_meta["original_direction"],
        )

    # --- Bypass: skip registration if metadata is identical ---
    if _metadata_matches(fixed_meta, moving_meta):
        print("  Registration BYPASS: identical spatial metadata detected")
        # Crop or pad moving volume to match fixed dimensions
        registered_hu = _match_dimensions(moving_hu, fixed_hu.shape)
        identity = sitk.Euler3DTransform()
        return registered_hu, identity, fixed_image

    moving_image = moving_meta.get("sitk_reference")
    if moving_image is None:
        moving_image = _numpy_to_sitk(
            moving_hu, moving_meta["spacing"],
            moving_meta["original_origin"], moving_meta["original_direction"],
        )

    # Initialise transform: align centres of geometry
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Configure registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Execute
    final_transform = registration.Execute(fixed_image, moving_image)

    # Apply transform to moving volume
    registered = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkBSpline,
        -1024.0,  # default pixel = air
        moving_image.GetPixelID(),
    )

    registered_hu = sitk.GetArrayFromImage(registered).astype(np.float32)
    return registered_hu, final_transform, fixed_image


def apply_transform_to_multichannel(
    moving_3ch: np.ndarray,
    transform: sitk.Transform,
    reference_image: sitk.Image,
    moving_meta: dict,
) -> np.ndarray:
    """
    Apply the rigid transform to a 3-channel volume, channel by channel.

    Parameters
    ----------
    moving_3ch : [D, 3, H, W] float32
    transform : from register_to_reference
    reference_image : sitk.Image defining the target space

    Returns
    -------
    registered_3ch : [D, 3, H, W] float32
    """
    D, C, H, W = moving_3ch.shape

    # If identity transform (bypass), just match dimensions
    if isinstance(transform, sitk.Euler3DTransform):
        params = transform.GetParameters()
        if all(p == 0.0 for p in params):
            print("  Multichannel transform BYPASS: identity transform")
            ref_size = reference_image.GetSize()  # (W, H, D) in sitk ordering
            target_d = ref_size[2]
            target_h = ref_size[1]
            target_w = ref_size[0]
            result = np.zeros((target_d, C, target_h, target_w), dtype=np.float32)
            d = min(D, target_d)
            h = min(H, target_h)
            w = min(W, target_w)
            result[:d, :, :h, :w] = moving_3ch[:d, :, :h, :w]
            return result

    registered_channels = []

    for c in range(C):
        channel_vol = moving_3ch[:, c, :, :]
        sitk_ch = sitk.GetImageFromArray(channel_vol.astype(np.float32))

        # Copy spatial info from the moving reference
        moving_ref = moving_meta.get("sitk_reference")
        if moving_ref is not None:
            sitk_ch.SetSpacing(moving_ref.GetSpacing())
            sitk_ch.SetOrigin(moving_ref.GetOrigin())
            sitk_ch.SetDirection(moving_ref.GetDirection())
        else:
            sp = moving_meta["spacing"]
            sitk_ch.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))

        resampled = sitk.Resample(
            sitk_ch,
            reference_image,
            transform,
            sitk.sitkBSpline,
            0.0,
            sitk_ch.GetPixelID(),
        )
        registered_channels.append(
            sitk.GetArrayFromImage(resampled).astype(np.float32)
        )

    return np.stack(registered_channels, axis=1)  # [D, 3, H, W]
