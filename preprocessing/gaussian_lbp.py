"""Gaussian filtering and single-scale uniform LBP image enhancement."""

import cv2
import numpy as np


def uniform_lbp(lightness, points=8, radius=1.0):
    """Return a rotation-invariant uniform LBP map.

    Uniform patterns are encoded by their number of one bits (0..P).
    Every non-uniform pattern is encoded as P + 1. This gives P + 2
    possible values, matching the common ``uniform`` LBP definition.
    """

    if lightness.ndim != 2:
        raise ValueError("LBP input must be a single-channel image.")

    if not isinstance(points, int) or points < 4:
        raise ValueError("LBP points must be an integer of at least four.")

    if points > 253:
        raise ValueError("LBP points must not exceed 253.")

    if radius <= 0:
        raise ValueError("LBP radius must be greater than zero.")

    height, width = lightness.shape
    grid_y, grid_x = np.indices(
        (height, width),
        dtype=np.float32,
    )
    centre = lightness.astype(np.float32)

    one_count = np.zeros(
        (height, width),
        dtype=np.uint16,
    )
    transition_count = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    first_bit = None
    previous_bit = None

    for point in range(points):
        angle = 2.0 * np.pi * point / points
        offset_x = np.float32(radius * np.cos(angle))
        offset_y = np.float32(radius * np.sin(angle))

        # OpenCV 5 requires both remapping arrays to be CV_32FC1.
        # Explicit conversion prevents NumPy scalar promotion to float64.
        sample_x = np.asarray(
            grid_x + offset_x,
            dtype=np.float32,
        )
        sample_y = np.asarray(
            grid_y - offset_y,
            dtype=np.float32,
        )

        neighbour = cv2.remap(
            centre,
            sample_x,
            sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        bit = neighbour >= centre

        one_count += bit

        if first_bit is None:
            first_bit = bit
        else:
            transition_count += bit != previous_bit

        previous_bit = bit

    transition_count += previous_bit != first_bit

    return np.where(
        transition_count <= 2,
        one_count,
        points + 1,
    ).astype(np.uint8)


def enhance_gaussian_lbp(
    image,
    gaussian_kernel=3,
    gaussian_sigma=0.8,
    lbp_points=8,
    lbp_radius=1.0,
    lbp_weight=0.25,
    return_maps=False,
):
    """Create a YOLO-compatible Gaussian-LBP enhanced BGR image.

    Gaussian filtering and LBP are applied to LAB lightness. The filtered
    lightness and the normalized LBP map are fused while the two original
    colour channels are retained. Image geometry is never changed, so the
    original YOLO bounding-box labels remain valid.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a three-channel BGR image.")

    if image.dtype != np.uint8:
        raise ValueError("Input image must use the uint8 data type.")

    if (
        not isinstance(gaussian_kernel, int)
        or gaussian_kernel <= 1
        or gaussian_kernel % 2 == 0
    ):
        raise ValueError(
            "Gaussian kernel must be an odd integer greater than one."
        )

    if gaussian_sigma <= 0:
        raise ValueError("Gaussian sigma must be greater than zero.")

    if not 0.0 <= lbp_weight <= 1.0:
        raise ValueError("LBP weight must be between zero and one.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    gaussian = cv2.GaussianBlur(
        lightness,
        (gaussian_kernel, gaussian_kernel),
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    lbp_codes = uniform_lbp(
        gaussian,
        points=lbp_points,
        radius=lbp_radius,
    )

    # Use a fixed scale rather than per-image min/max normalization so that
    # identical LBP codes always have the same intensity in every image.
    lbp_map = np.rint(
        lbp_codes.astype(np.float32)
        * (255.0 / (lbp_points + 1))
    ).astype(np.uint8)

    fused_lightness = cv2.addWeighted(
        gaussian,
        1.0 - lbp_weight,
        lbp_map,
        lbp_weight,
        0.0,
    )

    enhanced_lab = cv2.merge(
        (fused_lightness, channel_a, channel_b)
    )
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    if return_maps:
        return {
            "enhanced": enhanced,
            "original_lightness": lightness,
            "gaussian_lightness": gaussian,
            "lbp_codes": lbp_codes,
            "lbp_map": lbp_map,
            "fused_lightness": fused_lightness,
        }

    return enhanced
