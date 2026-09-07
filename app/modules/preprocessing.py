import cv2
import numpy as np

from config import (
    BILATERAL_DIAMETER,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    MORPH_KERNEL_SIZES,
    MORPH_ALPHA,
    MORPH_BETA,
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_bgr_image(image: np.ndarray) -> None:
    """
    Validate that the input is a valid three-channel BGR image.

    Parameters
    ----------
    image : np.ndarray
        Input image loaded using OpenCV.

    Raises
    ------
    ValueError
        If the image is None or is not a three-channel BGR image.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Input image must be a NumPy array."
        )

    if image.ndim != 3:
        raise ValueError(
            "Input image must contain three dimensions."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Input image must be a three-channel BGR image."
        )


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def validate_kernel_sizes(
    kernel_sizes
) -> None:
    """
    Validate morphology kernel sizes.

    Every kernel must be an odd integer greater than one.
    """

    if not kernel_sizes:
        raise ValueError(
            "At least one morphology kernel size is required."
        )

    for size in kernel_sizes:

        if not isinstance(size, int):
            raise ValueError(
                "Morphology kernel sizes must be integers."
            )

        if size <= 1 or size % 2 == 0:
            raise ValueError(
                "Morphology kernel sizes must be "
                "odd integers greater than one."
            )


# ============================================================
# CONVERT BGR TO LAB
# ============================================================

def convert_to_lab(
    image: np.ndarray
):
    """
    Convert a BGR PCB image to LAB colour space.

    Returns
    -------
    tuple
        LAB image together with the L, A and B channels.
    """

    validate_bgr_image(
        image
    )

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness, channel_a, channel_b = cv2.split(
        lab_image
    )

    return (
        lab_image,
        lightness,
        channel_a,
        channel_b,
    )


# ============================================================
# BILATERAL FILTERING
# ============================================================

def apply_bilateral_filter(
    lightness: np.ndarray,
    diameter: int = BILATERAL_DIAMETER,
    sigma_color: float = BILATERAL_SIGMA_COLOR,
    sigma_space: float = BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """
    Apply the final Technique D bilateral-filtering
    configuration to the LAB lightness channel.

    Final configuration:
        diameter = 3
        sigma colour = 15
        sigma space = 3
    """

    if lightness is None:
        raise ValueError(
            "Lightness channel cannot be None."
        )

    filtered_lightness = cv2.bilateralFilter(
        lightness,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return filtered_lightness


# ============================================================
# MULTI-SCALE MORPHOLOGICAL MAPS
# ============================================================

def create_multiscale_morphology_maps(
    filtered_lightness: np.ndarray,
    kernel_sizes=MORPH_KERNEL_SIZES,
):
    """
    Generate multi-scale Top-Hat and Black-Hat maps.

    An elliptical structuring element is created for each
    kernel size. The strongest response at every pixel is then
    retained using maximum fusion.

    Final kernel sizes:
        7, 15, 31

    Returns
    -------
    tuple
        multi_white : np.ndarray
            Maximum-fused Top-Hat response.

        multi_black : np.ndarray
            Maximum-fused Black-Hat response.

        white_maps : list[np.ndarray]
            Individual Top-Hat maps.

        black_maps : list[np.ndarray]
            Individual Black-Hat maps.
    """

    if filtered_lightness is None:
        raise ValueError(
            "Filtered lightness image cannot be None."
        )

    kernel_sizes = tuple(
        kernel_sizes
    )

    validate_kernel_sizes(
        kernel_sizes
    )

    white_maps = []
    black_maps = []

    for size in kernel_sizes:

        # ----------------------------------------------------
        # Elliptical structuring element
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )


        # ----------------------------------------------------
        # White Top-Hat
        # ----------------------------------------------------

        white_top_hat = cv2.morphologyEx(
            filtered_lightness,
            cv2.MORPH_TOPHAT,
            kernel,
        )

        white_maps.append(
            white_top_hat
        )


        # ----------------------------------------------------
        # Black-Hat
        # ----------------------------------------------------

        black_hat = cv2.morphologyEx(
            filtered_lightness,
            cv2.MORPH_BLACKHAT,
            kernel,
        )

        black_maps.append(
            black_hat
        )


    # --------------------------------------------------------
    # Maximum fusion across all morphology scales
    # --------------------------------------------------------

    multi_white = np.maximum.reduce(
        white_maps
    )

    multi_black = np.maximum.reduce(
        black_maps
    )


    return (
        multi_white,
        multi_black,
        white_maps,
        black_maps,
    )


# ============================================================
# MORPHOLOGICAL ENHANCEMENT
# ============================================================

def enhance_lightness_channel(
    filtered_lightness: np.ndarray,
    multi_white: np.ndarray,
    multi_black: np.ndarray,
    alpha: float = MORPH_ALPHA,
    beta: float = MORPH_BETA,
) -> np.ndarray:
    """
    Apply the final morphological enhancement equation.

    enhanced =
        filtered
        + alpha * Top-Hat
        - beta * Black-Hat

    Final weights:
        alpha = 0.7
        beta = 0.7
    """

    if filtered_lightness is None:
        raise ValueError(
            "Filtered lightness image cannot be None."
        )

    if multi_white is None:
        raise ValueError(
            "Top-Hat map cannot be None."
        )

    if multi_black is None:
        raise ValueError(
            "Black-Hat map cannot be None."
        )


    # Convert to float before weighted arithmetic.
    filtered_float = (
        filtered_lightness.astype(
            np.float32
        )
    )

    white_float = (
        multi_white.astype(
            np.float32
        )
    )

    black_float = (
        multi_black.astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # Technique D enhancement equation
    # --------------------------------------------------------

    enhanced_lightness = (
        filtered_float
        + alpha * white_float
        - beta * black_float
    )


    # --------------------------------------------------------
    # Maintain valid 8-bit image range
    # --------------------------------------------------------

    enhanced_lightness = np.clip(
        enhanced_lightness,
        0,
        255,
    ).astype(
        np.uint8
    )


    return enhanced_lightness


# ============================================================
# RECONSTRUCT FINAL BGR IMAGE
# ============================================================

def reconstruct_bgr_image(
    enhanced_lightness: np.ndarray,
    channel_a: np.ndarray,
    channel_b: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the final Technique D image.

    The enhanced L channel is merged with the original
    LAB A and B channels before conversion back to BGR.
    """

    enhanced_lab = cv2.merge(
        (
            enhanced_lightness,
            channel_a,
            channel_b,
        )
    )

    enhanced_bgr = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    return enhanced_bgr


# ============================================================
# COMPLETE TECHNIQUE D PIPELINE
# ============================================================

def apply_technique_d(
    image: np.ndarray,
    return_stages: bool = False,
):
    """
    Apply the complete final Technique D preprocessing pipeline.

    Processing sequence
    -------------------
    Original BGR image
        -> LAB colour conversion
        -> Bilateral Filtering on L channel
        -> Multi-scale Top-Hat
        -> Multi-scale Black-Hat
        -> Maximum morphology fusion
        -> Weighted enhancement
        -> LAB reconstruction
        -> BGR output

    Parameters
    ----------
    image : np.ndarray
        Original PCB image in BGR format.

    return_stages : bool
        When False, only the final enhanced image is returned.

        When True, intermediate processing stages are returned
        as a dictionary. This will later be useful for the
        Processing Analysis page in the GUI.

    Returns
    -------
    np.ndarray or dict
        Final enhanced BGR image, or a dictionary containing
        intermediate outputs.
    """

    validate_bgr_image(
        image
    )


    # --------------------------------------------------------
    # Preserve original image
    # --------------------------------------------------------

    original_image = image.copy()


    # --------------------------------------------------------
    # Step 1: BGR -> LAB
    # --------------------------------------------------------

    (
        _,
        lightness,
        channel_a,
        channel_b,
    ) = convert_to_lab(
        original_image
    )


    # --------------------------------------------------------
    # Step 2: Bilateral Filtering
    # --------------------------------------------------------

    filtered_lightness = apply_bilateral_filter(
        lightness,
        diameter=BILATERAL_DIAMETER,
        sigma_color=BILATERAL_SIGMA_COLOR,
        sigma_space=BILATERAL_SIGMA_SPACE,
    )


    # --------------------------------------------------------
    # Step 3: Multi-scale morphology
    # --------------------------------------------------------

    (
        multi_white,
        multi_black,
        white_maps,
        black_maps,
    ) = create_multiscale_morphology_maps(
        filtered_lightness,
        kernel_sizes=MORPH_KERNEL_SIZES,
    )


    # --------------------------------------------------------
    # Step 4: Weighted morphological enhancement
    # --------------------------------------------------------

    enhanced_lightness = enhance_lightness_channel(
        filtered_lightness,
        multi_white,
        multi_black,
        alpha=MORPH_ALPHA,
        beta=MORPH_BETA,
    )


    # --------------------------------------------------------
    # Step 5: Reconstruct colour image
    # --------------------------------------------------------

    enhanced_image = reconstruct_bgr_image(
        enhanced_lightness,
        channel_a,
        channel_b,
    )


    # --------------------------------------------------------
    # Return final image only
    # --------------------------------------------------------

    if not return_stages:
        return enhanced_image


    # --------------------------------------------------------
    # Return intermediate stages for GUI analysis
    # --------------------------------------------------------

    return {
        "original": original_image,

        "original_lightness":
            lightness,

        "filtered_lightness":
            filtered_lightness,

        "white_top_hat":
            multi_white,

        "black_hat":
            multi_black,

        "enhanced_lightness":
            enhanced_lightness,

        "enhanced":
            enhanced_image,

        "white_maps":
            white_maps,

        "black_maps":
            black_maps,
    }


# ============================================================
# DISPLAY FINAL CONFIGURATION
# ============================================================

def get_technique_d_configuration():
    """
    Return the final Technique D parameter configuration.

    Useful later for displaying the selected processing
    configuration inside the Streamlit GUI.
    """

    return {
        "Technique":
            (
                "Bilateral Filtering + "
                "Multi-scale Top-Hat/Black-Hat "
                "Morphological Enhancement"
            ),

        "Bilateral Diameter":
            BILATERAL_DIAMETER,

        "Sigma Colour":
            BILATERAL_SIGMA_COLOR,

        "Sigma Space":
            BILATERAL_SIGMA_SPACE,

        "Morphology Kernels":
            list(
                MORPH_KERNEL_SIZES
            ),

        "Morphology Kernel Shape":
            "Ellipse",

        "Alpha":
            MORPH_ALPHA,

        "Beta":
            MORPH_BETA,
    }