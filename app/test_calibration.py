from pathlib import Path

import cv2

from modules.calibration import (
    calibrate_image,
    calculate_bbox_physical_size,
)


# ============================================================
# TEST DATASET
# ============================================================

TEST_FOLDER = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_reduced"
    r"\test"
    r"\images"
)


def find_first_image(folder):

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    for path in sorted(
        folder.iterdir()
    ):

        if (
            path.is_file()
            and path.suffix.lower()
            in valid_extensions
        ):
            return path


    raise FileNotFoundError(
        "No test image found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PCB IMAGE CALIBRATION TEST"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_path = find_first_image(
        TEST_FOLDER
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:
        raise RuntimeError(
            "Failed to load test image."
        )


    height, width = (
        image.shape[:2]
    )


    print(
        f"\nTest image:\n{image_path}"
    )

    print(
        f"\nOriginal dimensions: "
        f"{width} x {height}"
    )


    # --------------------------------------------------------
    # Identity-style perspective points
    #
    # These use the four image corners so the output should
    # remain geometrically almost identical.
    # --------------------------------------------------------

    corner_points = [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]


    # Example physical PCB width for testing.
    pcb_width_mm = 100.0


    # --------------------------------------------------------
    # Run calibration
    # --------------------------------------------------------

    calibration = calibrate_image(
        image,
        corner_points=corner_points,
        pcb_width_mm=pcb_width_mm,
    )


    calibrated_image = (
        calibration["image"]
    )


    print(
        "\nPerspective rectification: "
        f"{calibration['rectification_applied']}"
    )


    print(
        "Calibrated dimensions: "
        f"{calibration['calibrated_width']} x "
        f"{calibration['calibrated_height']}"
    )


    print(
        "Known PCB width: "
        f"{calibration['pcb_width_mm']} mm"
    )


    print(
        "Spatial scale: "
        f"{calibration['mm_per_pixel']:.6f} mm/pixel"
    )


    # --------------------------------------------------------
    # Demonstrate bounding-box conversion
    # --------------------------------------------------------

    example_detection = {
        "width": 30.0,
        "height": 20.0,
    }


    physical_size = (
        calculate_bbox_physical_size(
            example_detection,
            calibration[
                "mm_per_pixel"
            ],
        )
    )


    print(
        "\nExample bounding box:"
    )

    print(
        f"Width : "
        f"{physical_size['width_pixels']:.1f} px "
        f"= {physical_size['width_mm']:.2f} mm"
    )

    print(
        f"Height: "
        f"{physical_size['height_pixels']:.1f} px "
        f"= {physical_size['height_mm']:.2f} mm"
    )


    # --------------------------------------------------------
    # Save calibrated image
    # --------------------------------------------------------

    output_folder = Path(
        "app/outputs/processed"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        output_folder
        / "test_calibrated.jpg"
    )


    cv2.imwrite(
        str(output_path),
        calibrated_image,
    )


    print(
        "\nCalibrated image saved to:"
    )

    print(
        output_path.resolve()
    )


    print(
        "\nCalibration test completed successfully."
    )


if __name__ == "__main__":
    main()