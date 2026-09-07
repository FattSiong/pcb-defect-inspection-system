import cv2
import numpy as np


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_calibration_image(image: np.ndarray) -> None:
    """
    Validate that the input image is a three-channel
    OpenCV BGR image.
    """

    if image is None:
        raise ValueError(
            "Calibration image cannot be None."
        )

    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Calibration image must be a NumPy array."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Calibration image must be a "
            "three-channel BGR image."
        )


# ============================================================
# ORDER FOUR CORNER POINTS
# ============================================================

def order_corner_points(points):
    """
    Arrange four coordinates as:

    Top-left
    Top-right
    Bottom-right
    Bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.shape != (4, 2):
        raise ValueError(
            "Exactly four corner points are required."
        )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    coordinate_sum = points.sum(
        axis=1
    )

    ordered[0] = points[
        np.argmin(
            coordinate_sum
        )
    ]

    ordered[2] = points[
        np.argmax(
            coordinate_sum
        )
    ]

    coordinate_difference = (
        np.diff(
            points,
            axis=1,
        )
        .reshape(-1)
    )

    ordered[1] = points[
        np.argmin(
            coordinate_difference
        )
    ]

    ordered[3] = points[
        np.argmax(
            coordinate_difference
        )
    ]

    return ordered


# ============================================================
# CHECK WHETHER CORNERS ARE INSIDE IMAGE
# ============================================================

def corner_points_inside_image(
    image,
    points,
):
    """
    Check whether all four PCB corner points are within
    the image boundaries.

    Returns
    -------
    bool
        True when every point is valid.
    """

    validate_calibration_image(
        image
    )

    try:

        ordered = order_corner_points(
            points
        )

    except ValueError:

        return False


    height, width = (
        image.shape[:2]
    )


    for x, y in ordered:

        if (
            x < 0
            or y < 0
            or x > width - 1
            or y > height - 1
        ):
            return False


    return True


# ============================================================
# VALIDATE CORNER POINTS
# ============================================================

def validate_corner_points(
    image,
    points,
):
    """
    Raise an error when detected PCB corners are invalid.
    """

    if not corner_points_inside_image(
        image,
        points,
    ):

        raise ValueError(
            "Detected PCB corner lies outside "
            "the image boundaries."
        )


# ============================================================
# AUTOMATIC PCB BOUNDARY DETECTION
# ============================================================

def detect_pcb_corners(
    image: np.ndarray,
    min_area_ratio: float = 0.15,
):
    """
    Automatically identify a large PCB boundary.

    Processing:
        BGR
        -> Grayscale
        -> Gaussian Blur
        -> Automatic Canny
        -> Morphological Closing
        -> Contour Search
        -> Quadrilateral Detection

    If a reliable four-sided region cannot be found,
    the function returns success=False so that the main
    inspection pipeline can safely use the original image.
    """

    validate_calibration_image(
        image
    )


    height, width = (
        image.shape[:2]
    )

    image_area = float(
        width * height
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    # --------------------------------------------------------
    # NOISE REDUCTION
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )


    # --------------------------------------------------------
    # AUTOMATIC CANNY THRESHOLDS
    # --------------------------------------------------------

    median_intensity = float(
        np.median(
            blurred
        )
    )


    lower_threshold = int(
        max(
            0,
            0.66 * median_intensity,
        )
    )


    upper_threshold = int(
        min(
            255,
            1.33 * median_intensity,
        )
    )


    if upper_threshold <= lower_threshold:

        lower_threshold = 50
        upper_threshold = 150


    edges = cv2.Canny(
        blurred,
        lower_threshold,
        upper_threshold,
    )


    # --------------------------------------------------------
    # CONNECT BROKEN EDGES
    # --------------------------------------------------------

    morphology_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )
    )


    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        morphology_kernel,
        iterations=2,
    )


    closed_edges = cv2.dilate(
        closed_edges,
        morphology_kernel,
        iterations=1,
    )


    # --------------------------------------------------------
    # FIND EXTERNAL CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )


    if not contours:

        return {
            "success": False,
            "points": None,
            "method": None,
            "area_ratio": 0.0,
            "score": 0.0,
            "edge_image": closed_edges,
            "message": (
                "No suitable PCB boundary "
                "could be detected."
            ),
        }


    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )


    candidates = []


    # --------------------------------------------------------
    # SEARCH FOR FOUR-SIDED PCB CONTOURS
    # --------------------------------------------------------

    for contour in contours:

        contour_area = float(
            cv2.contourArea(
                contour
            )
        )


        area_ratio = (
            contour_area
            / image_area
        )


        if area_ratio < min_area_ratio:
            continue


        perimeter = cv2.arcLength(
            contour,
            True,
        )


        for epsilon_factor in (
            0.015,
            0.020,
            0.025,
            0.030,
            0.040,
        ):

            approximation = (
                cv2.approxPolyDP(
                    contour,
                    epsilon_factor
                    * perimeter,
                    True,
                )
            )


            if len(approximation) != 4:
                continue


            if not cv2.isContourConvex(
                approximation
            ):
                continue


            points = (
                approximation
                .reshape(4, 2)
                .astype(np.float32)
            )


            ordered_points = (
                order_corner_points(
                    points
                )
            )


            # -----------------------------------------------
            # IMPORTANT:
            # reject candidate immediately when one or more
            # points fall outside the actual image.
            # -----------------------------------------------

            if not corner_points_inside_image(
                image,
                ordered_points,
            ):
                continue


            minimum_rectangle = (
                cv2.minAreaRect(
                    approximation
                )
            )


            rectangle_width = (
                minimum_rectangle[1][0]
            )

            rectangle_height = (
                minimum_rectangle[1][1]
            )


            rectangle_area = (
                rectangle_width
                * rectangle_height
            )


            if rectangle_area <= 0:

                rectangularity = 0.0

            else:

                rectangularity = min(
                    contour_area
                    / rectangle_area,
                    1.0,
                )


            score = (
                0.75
                * min(
                    area_ratio,
                    1.0,
                )
                + 0.25
                * rectangularity
            )


            candidates.append(
                {
                    "points":
                        ordered_points,

                    "area_ratio":
                        area_ratio,

                    "rectangularity":
                        rectangularity,

                    "score":
                        score,

                    "method":
                        "Quadrilateral Contour",
                }
            )


    # --------------------------------------------------------
    # USE BEST VALID QUADRILATERAL
    # --------------------------------------------------------

    if candidates:

        best_candidate = max(
            candidates,
            key=lambda item: (
                item["score"]
            ),
        )


        return {
            "success": True,

            "points":
                best_candidate[
                    "points"
                ],

            "method":
                best_candidate[
                    "method"
                ],

            "area_ratio":
                best_candidate[
                    "area_ratio"
                ],

            "score":
                best_candidate[
                    "score"
                ],

            "edge_image":
                closed_edges,

            "message":
                (
                    "PCB boundary detected "
                    "successfully."
                ),
        }


    # --------------------------------------------------------
    # MINIMUM-AREA RECTANGLE FALLBACK
    # --------------------------------------------------------
    #
    # Important:
    # cv2.minAreaRect() can produce rectangle points that
    # extend slightly outside the source image.
    #
    # We therefore validate the generated points before
    # reporting success.
    # --------------------------------------------------------

    for contour in contours:

        contour_area = float(
            cv2.contourArea(
                contour
            )
        )


        area_ratio = (
            contour_area
            / image_area
        )


        if area_ratio < min_area_ratio:
            continue


        minimum_rectangle = (
            cv2.minAreaRect(
                contour
            )
        )


        box_points = cv2.boxPoints(
            minimum_rectangle
        )


        ordered_points = (
            order_corner_points(
                box_points
            )
        )


        # -----------------------------------------------
        # Reject minimum-area rectangle if it extends
        # beyond the image.
        # -----------------------------------------------

        if not corner_points_inside_image(
            image,
            ordered_points,
        ):
            continue


        return {
            "success": True,

            "points":
                ordered_points,

            "method":
                "Minimum-Area Rectangle",

            "area_ratio":
                area_ratio,

            "score":
                area_ratio * 0.75,

            "edge_image":
                closed_edges,

            "message":
                (
                    "PCB boundary was approximated "
                    "using a minimum-area rectangle."
                ),
        }


    # --------------------------------------------------------
    # NO SAFE PCB BOUNDARY FOUND
    # --------------------------------------------------------

    return {
        "success": False,
        "points": None,
        "method": None,
        "area_ratio": 0.0,
        "score": 0.0,
        "edge_image": closed_edges,
        "message": (
            "A reliable PCB boundary could not be detected. "
            "The original image will be used instead."
        ),
    }


# ============================================================
# DRAW DETECTED PCB CORNERS
# ============================================================

def draw_detected_corners(
    image,
    points,
):
    """
    Draw detected PCB outline and four corner markers.
    """

    validate_calibration_image(
        image
    )


    if not corner_points_inside_image(
        image,
        points,
    ):

        raise ValueError(
            "Cannot preview invalid PCB corner points."
        )


    ordered_points = (
        order_corner_points(
            points
        )
    )


    preview = image.copy()


    integer_points = (
        ordered_points
        .astype(np.int32)
    )


    # --------------------------------------------------------
    # PCB OUTLINE
    # --------------------------------------------------------

    cv2.polylines(
        preview,
        [
            integer_points.reshape(
                (-1, 1, 2)
            )
        ],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )


    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]


    # --------------------------------------------------------
    # CORNER MARKERS
    # --------------------------------------------------------

    for point, label in zip(
        integer_points,
        labels,
    ):

        x = int(
            point[0]
        )

        y = int(
            point[1]
        )


        cv2.circle(
            preview,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )


        text_x = max(
            0,
            min(
                x + 10,
                preview.shape[1] - 40,
            ),
        )


        text_y = max(
            20,
            min(
                y - 10,
                preview.shape[0] - 5,
            ),
        )


        cv2.putText(
            preview,
            label,
            (
                text_x,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


    return preview


# ============================================================
# PERSPECTIVE RECTIFICATION
# ============================================================

def rectify_perspective(
    image,
    points,
):
    """
    Apply four-point perspective correction.
    """

    validate_calibration_image(
        image
    )


    validate_corner_points(
        image,
        points,
    )


    ordered_points = (
        order_corner_points(
            points
        )
    )


    (
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ) = ordered_points


    # --------------------------------------------------------
    # OUTPUT WIDTH
    # --------------------------------------------------------

    bottom_width = np.linalg.norm(
        bottom_right
        - bottom_left
    )


    top_width = np.linalg.norm(
        top_right
        - top_left
    )


    output_width = int(
        round(
            max(
                bottom_width,
                top_width,
            )
        )
    )


    # --------------------------------------------------------
    # OUTPUT HEIGHT
    # --------------------------------------------------------

    right_height = np.linalg.norm(
        bottom_right
        - top_right
    )


    left_height = np.linalg.norm(
        bottom_left
        - top_left
    )


    output_height = int(
        round(
            max(
                right_height,
                left_height,
            )
        )
    )


    if (
        output_width < 2
        or output_height < 2
    ):

        raise ValueError(
            "Detected PCB boundary produces "
            "an invalid rectification area."
        )


    destination_points = np.array(
        [
            [0, 0],

            [
                output_width - 1,
                0,
            ],

            [
                output_width - 1,
                output_height - 1,
            ],

            [
                0,
                output_height - 1,
            ],
        ],
        dtype=np.float32,
    )


    transformation_matrix = (
        cv2.getPerspectiveTransform(
            ordered_points,
            destination_points,
        )
    )


    rectified_image = (
        cv2.warpPerspective(
            image,
            transformation_matrix,
            (
                output_width,
                output_height,
            ),
        )
    )


    return (
        rectified_image,
        transformation_matrix,
        ordered_points,
    )


# ============================================================
# SPATIAL SCALE CALIBRATION
# ============================================================

def calculate_spatial_scale(
    image_width_pixels,
    pcb_width_mm,
):
    """
    Calculate millimetres represented by one pixel.
    """

    if image_width_pixels <= 0:
        raise ValueError(
            "Image width must be greater than zero."
        )


    if pcb_width_mm <= 0:
        raise ValueError(
            "Physical PCB width must be greater than zero."
        )


    return (
        float(pcb_width_mm)
        / float(image_width_pixels)
    )


# ============================================================
# PIXEL TO MILLIMETRES
# ============================================================

def pixels_to_millimetres(
    pixel_distance,
    mm_per_pixel,
):
    """
    Convert pixel distance into millimetres.
    """

    if pixel_distance < 0:
        raise ValueError(
            "Pixel distance cannot be negative."
        )


    if mm_per_pixel <= 0:
        raise ValueError(
            "Spatial scale must be greater than zero."
        )


    return (
        float(pixel_distance)
        * float(mm_per_pixel)
    )


# ============================================================
# BOUNDING-BOX PHYSICAL SIZE
# ============================================================

def calculate_bbox_physical_size(
    detection,
    mm_per_pixel,
):
    """
    Calculate approximate YOLO bounding-box dimensions
    in millimetres.
    """

    width_pixels = float(
        detection["width"]
    )


    height_pixels = float(
        detection["height"]
    )


    return {
        "width_pixels":
            width_pixels,

        "height_pixels":
            height_pixels,

        "width_mm":
            pixels_to_millimetres(
                width_pixels,
                mm_per_pixel,
            ),

        "height_mm":
            pixels_to_millimetres(
                height_pixels,
                mm_per_pixel,
            ),
    }


# ============================================================
# COMPLETE CALIBRATION PIPELINE
# ============================================================

def calibrate_image(
    image,
    corner_points=None,
    pcb_width_mm=None,
):
    """
    Apply optional perspective rectification and optional
    spatial scale calibration.

    When corner_points is None, the original image is
    retained unchanged.
    """

    validate_calibration_image(
        image
    )


    original_height, original_width = (
        image.shape[:2]
    )


    calibrated_image = (
        image.copy()
    )


    transformation_matrix = None
    ordered_points = None
    rectification_applied = False


    # --------------------------------------------------------
    # OPTIONAL RECTIFICATION
    # --------------------------------------------------------

    if corner_points is not None:

        (
            calibrated_image,
            transformation_matrix,
            ordered_points,
        ) = rectify_perspective(
            image,
            corner_points,
        )


        rectification_applied = True


    calibrated_height, calibrated_width = (
        calibrated_image.shape[:2]
    )


    # --------------------------------------------------------
    # OPTIONAL SPATIAL SCALE
    # --------------------------------------------------------

    mm_per_pixel = None


    if pcb_width_mm is not None:

        mm_per_pixel = (
            calculate_spatial_scale(
                calibrated_width,
                pcb_width_mm,
            )
        )


    return {
        "image":
            calibrated_image,

        "rectification_applied":
            rectification_applied,

        "original_width":
            original_width,

        "original_height":
            original_height,

        "calibrated_width":
            calibrated_width,

        "calibrated_height":
            calibrated_height,

        "pcb_width_mm":
            pcb_width_mm,

        "mm_per_pixel":
            mm_per_pixel,

        "transformation_matrix":
            transformation_matrix,

        "ordered_points":
            ordered_points,
    }