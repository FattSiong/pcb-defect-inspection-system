from pathlib import Path

import cv2

from modules.preprocessing import (
    apply_technique_d,
    get_technique_d_configuration,
)


# ============================================================
# TEST IMAGE
# ============================================================

TEST_IMAGE = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced\test\images"
)


def find_first_image(folder):

    extensions = {
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
            in extensions
        ):
            return path

    raise FileNotFoundError(
        "No test image found."
    )


def main():

    print("=" * 80)
    print(
        "TECHNIQUE D PREPROCESSING TEST"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    configuration = (
        get_technique_d_configuration()
    )

    print(
        "\nFinal configuration:"
    )

    for key, value in (
        configuration.items()
    ):

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Load test image
    # --------------------------------------------------------

    image_path = (
        find_first_image(
            TEST_IMAGE
        )
    )

    print(
        f"\nTest image:\n"
        f"{image_path}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            "Failed to load test image."
        )


    # --------------------------------------------------------
    # Apply Technique D
    # --------------------------------------------------------

    stages = apply_technique_d(
        image,
        return_stages=True,
    )


    print(
        "\nPreprocessing completed successfully."
    )

    print(
        f"Original shape: "
        f"{stages['original'].shape}"
    )

    print(
        f"Enhanced shape: "
        f"{stages['enhanced'].shape}"
    )


    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    output_dir = Path(
        "app/outputs/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    cv2.imwrite(
        str(
            output_dir
            / "test_original.jpg"
        ),
        stages["original"],
    )


    cv2.imwrite(
        str(
            output_dir
            / "test_bilateral_lightness.jpg"
        ),
        stages[
            "filtered_lightness"
        ],
    )


    cv2.imwrite(
        str(
            output_dir
            / "test_white_top_hat.jpg"
        ),
        stages[
            "white_top_hat"
        ],
    )


    cv2.imwrite(
        str(
            output_dir
            / "test_black_hat.jpg"
        ),
        stages[
            "black_hat"
        ],
    )


    cv2.imwrite(
        str(
            output_dir
            / "test_enhanced.jpg"
        ),
        stages[
            "enhanced"
        ],
    )


    print(
        "\nOutputs saved to:"
    )

    print(
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()