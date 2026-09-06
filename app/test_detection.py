from pathlib import Path

import cv2

from modules.preprocessing import (
    apply_technique_d,
)

from modules.detection import (
    detect_defects,
    draw_detections,
    summarise_detections,
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

    for image_path in sorted(
        folder.iterdir()
    ):

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in valid_extensions
        ):
            return image_path


    raise FileNotFoundError(
        "No test image was found."
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 80)
    print(
        "TECHNIQUE D COMPLETE DETECTION TEST"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Find PCB test image
    # --------------------------------------------------------

    image_path = find_first_image(
        TEST_FOLDER
    )


    print(
        f"\nTest image:\n{image_path}"
    )


    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    original_image = cv2.imread(
        str(image_path)
    )


    if original_image is None:

        raise RuntimeError(
            "Failed to read test image."
        )


    print(
        f"\nOriginal image shape: "
        f"{original_image.shape}"
    )


    # --------------------------------------------------------
    # Technique D preprocessing
    # --------------------------------------------------------

    processed_image = (
        apply_technique_d(
            original_image
        )
    )


    print(
        "Technique D preprocessing: SUCCESS"
    )


    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    detections, _ = (
        detect_defects(
            processed_image
        )
    )


    print(
        "YOLO detection: SUCCESS"
    )


    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    print(
        f"\nTotal detections: "
        f"{len(detections)}"
    )


    if detections:

        print(
            "\nDetected defects:"
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            print(
                f"\nDetection {index}"
            )

            print(
                f"  Class      : "
                f"{detection['class_name']}"
            )

            print(
                f"  Confidence : "
                f"{detection['confidence']:.4f}"
            )

            print(
                f"  Bounding box: "
                f"("
                f"{detection['x1']:.1f}, "
                f"{detection['y1']:.1f}, "
                f"{detection['x2']:.1f}, "
                f"{detection['y2']:.1f}"
                f")"
            )

    else:

        print(
            "\nNo defects detected."
        )


    # --------------------------------------------------------
    # Defect summary
    # --------------------------------------------------------

    summary = (
        summarise_detections(
            detections
        )
    )


    print(
        "\nDefect summary:"
    )


    for class_name, count in (
        summary.items()
    ):

        print(
            f"  {class_name:<20}: "
            f"{count}"
        )


    # --------------------------------------------------------
    # Draw detections on ORIGINAL IMAGE
    # --------------------------------------------------------

    annotated_image = (
        draw_detections(
            original_image,
            detections,
        )
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_folder = Path(
        "app/outputs/detections"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    processed_path = (
        output_folder
        / "test_processed.jpg"
    )


    annotated_path = (
        output_folder
        / "test_detection.jpg"
    )


    cv2.imwrite(
        str(processed_path),
        processed_image,
    )


    cv2.imwrite(
        str(annotated_path),
        annotated_image,
    )


    print(
        "\nResults saved to:"
    )

    print(
        processed_path.resolve()
    )

    print(
        annotated_path.resolve()
    )


    print(
        "\nComplete pipeline test finished."
    )


if __name__ == "__main__":
    main()