from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import time

import cv2

from bilateral_morphology import enhance_pcb


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def get_image_files(directory):
    """Return supported image files in a directory."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def process_split(
    source_root,
    output_root,
    split,
    bilateral_d,
    sigma_color,
    sigma_space,
    kernel_sizes,
    alpha,
    beta,
):
    """Process one dataset split and copy its YOLO labels."""

    source_images = source_root / split / "images"
    source_labels = source_root / split / "labels"

    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"

    if not source_labels.exists():
        raise FileNotFoundError(
            f"Label directory not found: {source_labels}"
        )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = get_image_files(source_images)

    if not image_files:
        raise RuntimeError(
            f"No images found in: {source_images}"
        )

    print()
    print("=" * 70)
    print(f"Processing split: {split}")
    print(f"Images found   : {len(image_files)}")
    print("=" * 70)

    start_time = time.perf_counter()

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):
        label_path = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}: "
                f"{label_path}"
            )

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if original is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        enhanced = enhance_pcb(
            original,
            bilateral_d=bilateral_d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            kernel_sizes=kernel_sizes,
            alpha=alpha,
            beta=beta,
        )

        # The enhancement must not change image geometry.
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

        # Use PNG to avoid adding JPEG compression artefacts.
        output_image_path = (
            output_images /
            f"{image_path.stem}.png"
        )

        output_label_path = (
            output_labels /
            f"{image_path.stem}.txt"
        )

        saved = cv2.imwrite(
            str(output_image_path),
            enhanced,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ],
        )

        if not saved:
            raise RuntimeError(
                f"Could not save image: {output_image_path}"
            )

        # Labels remain valid because images are not resized
        # or geometrically transformed.
        shutil.copy2(
            label_path,
            output_label_path,
        )

        if index % 100 == 0 or index == len(image_files):
            print(
                f"{split}: {index}/{len(image_files)}"
            )

    elapsed = time.perf_counter() - start_time

    print(f"Completed       : {split}")
    print(f"Elapsed time    : {elapsed:.2f} seconds")
    print(
        f"Average time    : "
        f"{elapsed / len(image_files):.4f} seconds/image"
    )

    return {
        "split": split,
        "images": len(image_files),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(image_files),
    }


def verify_split(output_root, split):
    """Verify that every output image has one label."""

    image_directory = output_root / split / "images"
    label_directory = output_root / split / "labels"

    image_files = get_image_files(image_directory)

    label_files = sorted(
        label_directory.glob("*.txt")
    )

    image_stems = {
        path.stem
        for path in image_files
    }

    label_stems = {
        path.stem
        for path in label_files
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        raise RuntimeError(
            f"{split} has images without labels: "
            f"{sorted(missing_labels)[:10]}"
        )

    if missing_images:
        raise RuntimeError(
            f"{split} has labels without images: "
            f"{sorted(missing_images)[:10]}"
        )

    print(
        f"{split:<6} "
        f"images={len(image_files)}, "
        f"labels={len(label_files)}"
    )

    return {
        "images": len(image_files),
        "labels": len(label_files),
    }


def create_data_yaml(output_root, splits):
    """Create an Ultralytics YOLO data.yaml file."""

    lines = [
        f'path: "{output_root.resolve().as_posix()}"',
        "",
    ]

    if "train" in splits:
        lines.append("train: train/images")

    if "val" in splits:
        lines.append("val: val/images")

    if "test" in splits:
        lines.append("test: test/images")

    lines.extend(
        [
            "",
            "names:",
        ]
    )

    for class_id, class_name in CLASS_NAMES.items():
        lines.append(
            f"  {class_id}: {class_name}"
        )

    yaml_path = output_root / "data.yaml"

    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Created YAML: {yaml_path}")


def save_processing_config(
    source_root,
    output_root,
    splits,
    bilateral_d,
    sigma_color,
    sigma_space,
    kernel_sizes,
    alpha,
    beta,
    statistics,
):
    """Save parameters and processing-time results."""

    configuration = {
        "source_dataset": str(source_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "splits": list(splits),
        "image_format": "png",
        "bilateral_filter": {
            "diameter": bilateral_d,
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
        },
        "morphology": {
            "kernel_shape": "ellipse",
            "kernel_sizes": list(kernel_sizes),
            "alpha": alpha,
            "beta": beta,
        },
        "processing_statistics": statistics,
    }

    config_path = (
        output_root /
        "processing_config.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(f"Created configuration: {config_path}")


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Create a PCB dataset enhanced using bilateral "
            "filtering and multi-scale morphology."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root directory of the source YOLO dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory for the processed dataset.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Dataset splits to process.",
    )

    parser.add_argument(
        "--bilateral-d",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sigma-color",
        type=float,
        default=25,
    )

    parser.add_argument(
        "--sigma-space",
        type=float,
        default=5,
    )

    parser.add_argument(
        "--kernels",
        type=int,
        nargs="+",
        default=[5, 11, 21],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow files in an existing output directory "
            "to be replaced."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_root}"
        )

    if source_root == output_root:
        raise ValueError(
            "Source and output directories cannot be the same."
        )

    if output_root.exists():
        contains_files = any(
            output_root.iterdir()
        )

        if contains_files and not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}\n"
                "Choose a new directory or add --overwrite."
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    kernel_sizes = tuple(args.kernels)

    if any(
        size <= 1 or size % 2 == 0
        for size in kernel_sizes
    ):
        raise ValueError(
            "Every morphology kernel must be an odd "
            "integer greater than one."
        )

    print("=" * 70)
    print("BILATERAL + MULTI-SCALE MORPHOLOGY DATASET")
    print("=" * 70)
    print(f"Source       : {source_root}")
    print(f"Output       : {output_root}")
    print(f"Splits       : {args.splits}")
    print(f"Bilateral d  : {args.bilateral_d}")
    print(f"Sigma colour : {args.sigma_color}")
    print(f"Sigma space  : {args.sigma_space}")
    print(f"Kernels      : {kernel_sizes}")
    print(f"Alpha        : {args.alpha}")
    print(f"Beta         : {args.beta}")

    statistics = []

    for split in args.splits:
        result = process_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            bilateral_d=args.bilateral_d,
            sigma_color=args.sigma_color,
            sigma_space=args.sigma_space,
            kernel_sizes=kernel_sizes,
            alpha=args.alpha,
            beta=args.beta,
        )

        statistics.append(result)

    print()
    print("=" * 70)
    print("VERIFYING PROCESSED DATASET")
    print("=" * 70)

    for split in args.splits:
        verify_split(
            output_root,
            split,
        )

    create_data_yaml(
        output_root,
        args.splits,
    )

    save_processing_config(
        source_root=source_root,
        output_root=output_root,
        splits=args.splits,
        bilateral_d=args.bilateral_d,
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        kernel_sizes=kernel_sizes,
        alpha=args.alpha,
        beta=args.beta,
        statistics=statistics,
    )

    print()
    print("=" * 70)
    print("DATASET PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Processed dataset: {output_root}")


if __name__ == "__main__":
    main()