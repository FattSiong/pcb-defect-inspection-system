import cv2
import numpy as np


def enhance_pcb(
    image,
    bilateral_d=5,
    sigma_color=25,
    sigma_space=5,
    kernel_sizes=(5, 11, 21),
    alpha=0.7,
    beta=0.7,
    return_maps=False,
):
    """Enhance PCB defects without changing image geometry."""

    if image is None:
        raise ValueError("Input image cannot be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a three-channel BGR image.")

    if any(size <= 1 or size % 2 == 0 for size in kernel_sizes):
        raise ValueError("Kernel sizes must be odd integers greater than one.")

    # OpenCV reads colour images in BGR format.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    filtered = cv2.bilateralFilter(
        lightness,
        d=bilateral_d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    white_maps = []
    black_maps = []

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )

        white_maps.append(
            cv2.morphologyEx(
                filtered,
                cv2.MORPH_TOPHAT,
                kernel,
            )
        )

        black_maps.append(
            cv2.morphologyEx(
                filtered,
                cv2.MORPH_BLACKHAT,
                kernel,
            )
        )

    multi_white = np.maximum.reduce(white_maps)
    multi_black = np.maximum.reduce(black_maps)

    enhanced_lightness = np.clip(
        filtered.astype(np.float32)
        + alpha * multi_white.astype(np.float32)
        - beta * multi_black.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    enhanced_lab = cv2.merge(
        (enhanced_lightness, channel_a, channel_b)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    if return_maps:
        return {
            "enhanced": enhanced_image,
            "filtered_lightness": filtered,
            "white_top_hat": multi_white,
            "black_hat": multi_black,
        }

    return enhanced_image