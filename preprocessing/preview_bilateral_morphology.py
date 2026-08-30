from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from bilateral_morphology import enhance_pcb


def save_image(path, image):
    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Could not save image: {path}")


def main():
    parser = ArgumentParser(
        description="Test bilateral and morphology enhancement on one PCB image."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to one PCB image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/bilateral_morphology_preview"),
        help="Directory for preview outputs.",
    )

    args = parser.parse_args()

    # Load the image.
    original = cv2.imread(str(args.image))

    if original is None:
        raise FileNotFoundError(
            f"Could not read image: {args.image}"
        )

    # Run the enhancement and request diagnostic maps.
    results = enhance_pcb(
        original,
        return_maps=True,
    )

    enhanced = results["enhanced"]

    # Basic correctness checks.
    assert enhanced.shape == original.shape, (
        f"Shape changed from {original.shape} "
        f"to {enhanced.shape}"
    )

    assert enhanced.dtype == original.dtype, (
        f"Data type changed from {original.dtype} "
        f"to {enhanced.dtype}"
    )

    assert enhanced.dtype == np.uint8, (
        f"Expected uint8, received {enhanced.dtype}"
    )

    assert enhanced.min() >= 0
    assert enhanced.max() <= 255

    # Ensure processing produced a change.
    mean_difference = np.mean(
        cv2.absdiff(original, enhanced)
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        args.output / "01_original.png",
        original,
    )

    save_image(
        args.output / "02_filtered_lightness.png",
        results["filtered_lightness"],
    )

    save_image(
        args.output / "03_white_top_hat.png",
        results["white_top_hat"],
    )

    save_image(
        args.output / "04_black_hat.png",
        results["black_hat"],
    )

    save_image(
        args.output / "05_enhanced.png",
        enhanced,
    )

    print("Single-image test passed.")
    print(f"Original shape : {original.shape}")
    print(f"Enhanced shape : {enhanced.shape}")
    print(f"Original dtype : {original.dtype}")
    print(f"Enhanced dtype : {enhanced.dtype}")
    print(f"Value range    : {enhanced.min()}–{enhanced.max()}")
    print(f"Mean difference: {mean_difference:.3f}")
    print(f"Outputs saved  : {args.output.resolve()}")


if __name__ == "__main__":
    main()