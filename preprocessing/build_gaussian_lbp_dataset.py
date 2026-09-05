"""Build a YOLO dataset enhanced with Gaussian filtering and LBP."""

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import time

import cv2

from gaussian_lbp import enhance_gaussian_lbp


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def get_image_files(directory):
    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory}"
        )

    image_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    stems = [path.stem for path in image_files]

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            "The source image directory contains duplicate file stems. "
            "They cannot be safely converted to PNG."
        )

    return image_files


def process_split(
    source_root,
    output_root,
    split,
    gaussian_kernel,
    gaussian_sigma,
    lbp_points,
    lbp_radius,
    lbp_weight,
):
    """Process one split and copy the corresponding YOLO labels."""

    source_images = source_root / split / "images"
    source_labels = source_root / split / "labels"
    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"

    if not source_labels.exists():
        raise FileNotFoundError(
            f"Label directory not found: {source_labels}"
        )

    image_files = get_image_files(source_images)

    if not image_files:
        raise RuntimeError(f"No images found in: {source_images}")

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"Processing split: {split}")
    print(f"Images found   : {len(image_files)}")
    print("=" * 72)

    start_time = time.perf_counter()

    for index, image_path in enumerate(image_files, start=1):
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}: {label_path}"
            )

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if original is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        enhanced = enhance_gaussian_lbp(
            original,
            gaussian_kernel=gaussian_kernel,
            gaussian_sigma=gaussian_sigma,
            lbp_points=lbp_points,
            lbp_radius=lbp_radius,
            lbp_weight=lbp_weight,
        )

        if enhanced.shape != original.shape:
            raise RuntimeError(
                f"Image shape changed for {image_path.name}: "
                f"{original.shape} -> {enhanced.shape}"
            )

        if enhanced.dtype != original.dtype:
            raise RuntimeError(
                f"Image type changed for {image_path.name}: "
                f"{original.dtype} -> {enhanced.dtype}"
            )

        output_image_path = output_images / f"{image_path.stem}.png"
        output_label_path = output_labels / f"{image_path.stem}.txt"

        saved = cv2.imwrite(
            str(output_image_path),
            enhanced,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        if not saved:
            raise RuntimeError(
                f"Could not save image: {output_image_path}"
            )

        # Gaussian filtering and LBP fusion do not change geometry.
        shutil.copy2(label_path, output_label_path)

        if index % 100 == 0 or index == len(image_files):
            print(f"{split}: {index}/{len(image_files)}")

    elapsed = time.perf_counter() - start_time

    print(f"Completed       : {split}")
    print(f"Elapsed time    : {elapsed:.2f} seconds")
    print(
        "Average time    : "
        f"{elapsed / len(image_files):.4f} seconds/image"
    )

    return {
        "split": split,
        "images": len(image_files),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(image_files),
    }


def verify_split(output_root, split):
    image_directory = output_root / split / "images"
    label_directory = output_root / split / "labels"

    image_files = get_image_files(image_directory)
    label_files = sorted(label_directory.glob("*.txt"))
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels or missing_images:
        raise RuntimeError(
            f"{split} verification failed. "
            f"Images without labels: {missing_labels[:10]}; "
            f"labels without images: {missing_images[:10]}"
        )

    print(
        f"{split:<6} images={len(image_files)}, "
        f"labels={len(label_files)}"
    )


def create_data_yaml(output_root, splits):
    lines = [
        f'path: "{output_root.resolve().as_posix()}"',
        "",
    ]

    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: {split}/images")

    lines.extend(["", "names:"])

    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {class_name}")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Created YAML    : {yaml_path}")


def save_processing_config(
    source_root,
    output_root,
    splits,
    gaussian_kernel,
    gaussian_sigma,
    lbp_points,
    lbp_radius,
    lbp_weight,
    statistics,
):
    configuration = {
        "method": "gaussian_uniform_lbp_luminance_fusion",
        "source_dataset": str(source_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "splits": list(splits),
        "image_format": "png",
        "gaussian_filter": {
            "kernel_size": gaussian_kernel,
            "sigma": gaussian_sigma,
            "colour_space": "LAB lightness",
        },
        "lbp": {
            "points": lbp_points,
            "radius": lbp_radius,
            "method": "rotation-invariant uniform",
            "number_of_codes": lbp_points + 2,
        },
        "fusion": {
            "gaussian_weight": 1.0 - lbp_weight,
            "lbp_weight": lbp_weight,
            "preserved_channels": ["LAB A", "LAB B"],
        },
        "processing_statistics": statistics,
    }

    config_path = output_root / "processing_config.json"
    config_path.write_text(
        json.dumps(configuration, indent=4),
        encoding="utf-8",
    )
    print(f"Created config  : {config_path}")


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Create a YOLO dataset using Gaussian filtering and "
            "single-scale uniform LBP luminance fusion."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root of the source YOLO dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root for the generated YOLO dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Splits to generate; omit test during parameter tuning.",
    )
    parser.add_argument(
        "--gaussian-kernel",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--lbp-points",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lbp-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lbp-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow matching files in a non-empty output to be replaced.",
    )
    return parser.parse_args()


def validate_arguments(args):
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_root}"
        )

    if source_root == output_root:
        raise ValueError("Source and output datasets cannot be the same.")

    if (
        args.gaussian_kernel <= 1
        or args.gaussian_kernel % 2 == 0
    ):
        raise ValueError(
            "Gaussian kernel must be an odd integer greater than one."
        )

    if args.gaussian_sigma <= 0:
        raise ValueError("Gaussian sigma must be greater than zero.")

    if args.lbp_points < 4 or args.lbp_points > 253:
        raise ValueError("LBP points must be between 4 and 253.")

    if args.lbp_radius <= 0:
        raise ValueError("LBP radius must be greater than zero.")

    if not 0.0 <= args.lbp_weight <= 1.0:
        raise ValueError("LBP weight must be between zero and one.")

    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}\n"
            "Choose a new output or add --overwrite."
        )

    return source_root, output_root


def main():
    args = parse_arguments()
    source_root, output_root = validate_arguments(args)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GAUSSIAN + SINGLE-SCALE UNIFORM LBP DATASET")
    print("=" * 72)
    print(f"Source          : {source_root}")
    print(f"Output          : {output_root}")
    print(f"Splits          : {args.splits}")
    print(f"Gaussian kernel : {args.gaussian_kernel}")
    print(f"Gaussian sigma  : {args.gaussian_sigma}")
    print(f"LBP points      : {args.lbp_points}")
    print(f"LBP radius      : {args.lbp_radius}")
    print(f"LBP weight      : {args.lbp_weight}")

    statistics = []

    for split in args.splits:
        statistics.append(
            process_split(
                source_root=source_root,
                output_root=output_root,
                split=split,
                gaussian_kernel=args.gaussian_kernel,
                gaussian_sigma=args.gaussian_sigma,
                lbp_points=args.lbp_points,
                lbp_radius=args.lbp_radius,
                lbp_weight=args.lbp_weight,
            )
        )

    print()
    print("=" * 72)
    print("VERIFYING GENERATED DATASET")
    print("=" * 72)

    for split in args.splits:
        verify_split(output_root, split)

    create_data_yaml(output_root, args.splits)
    save_processing_config(
        source_root=source_root,
        output_root=output_root,
        splits=args.splits,
        gaussian_kernel=args.gaussian_kernel,
        gaussian_sigma=args.gaussian_sigma,
        lbp_points=args.lbp_points,
        lbp_radius=args.lbp_radius,
        lbp_weight=args.lbp_weight,
        statistics=statistics,
    )

    print()
    print("=" * 72)
    print("DATASET GENERATION COMPLETED")
    print("=" * 72)
    print(f"Dataset: {output_root}")
    print(f"YAML   : {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()

