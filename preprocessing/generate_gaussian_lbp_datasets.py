"""Generate Gaussian + LBP datasets for a controlled parameter search.

The image geometry is not changed, so the original YOLO labels can be copied
without modifying their normalised bounding-box coordinates.
"""

from itertools import product
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# =============================================================================
# EDIT THESE SETTINGS BEFORE RUNNING
# =============================================================================

SOURCE_DATASET = Path(
    r"C:\Users\ASUS\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\ASUS\Downloads\pcb_dataset_gaussian_lbp"
)

# Only train and validation data are generated during parameter tuning.
SPLITS = ("train", "val")

# Set this to a small integer, such as 10, for a quick trial run.
# Keep it as None when generating the real tuning datasets.
MAX_IMAGES_PER_SPLIT = None

# Missing labels normally indicate a dataset problem, so generation stops.
STRICT_LABELS = True

GAUSSIAN_CONFIGS = [
    {"name": "g3_s05", "kernel_size": 3, "sigma": 0.5},
    {"name": "g5_s10", "kernel_size": 5, "sigma": 1.0},
    {"name": "g7_s15", "kernel_size": 7, "sigma": 1.5},
]

LBP_CONFIGS = [
    {"name": "p8_r1", "points": 8, "radius": 1},
    {"name": "p16_r2", "points": 16, "radius": 2},
    {"name": "p24_r3", "points": 24, "radius": 3},
]

LBP_METHOD = "uniform"

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_image(image, gaussian_config, lbp_config):
    """Return a three-channel [Gaussian, LBP, Gaussian] image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = gaussian_config["kernel_size"]
    sigma = gaussian_config["sigma"]

    filtered = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    points = lbp_config["points"]
    radius = lbp_config["radius"]

    lbp = local_binary_pattern(
        filtered,
        P=points,
        R=radius,
        method=LBP_METHOD,
    )

    # Uniform LBP produces codes in the fixed range 0 to P + 1.
    # Fixed scaling keeps the same code-to-intensity mapping for all images
    # produced by one configuration.
    lbp_u8 = np.rint(
        lbp * 255.0 / (points + 1)
    ).clip(0, 255).astype(np.uint8)

    # This symmetric BGR representation is unchanged by BGR/RGB reversal.
    return cv2.merge([filtered, lbp_u8, filtered])


# =============================================================================
# DATASET UTILITIES
# =============================================================================

def configuration_name(gaussian_config, lbp_config):
    return f"{gaussian_config['name']}_{lbp_config['name']}"


def get_image_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if MAX_IMAGES_PER_SPLIT is not None:
        files = files[:MAX_IMAGES_PER_SPLIT]

    return files


def validate_settings():
    if SOURCE_DATASET.resolve() == OUTPUT_ROOT.resolve():
        raise ValueError("SOURCE_DATASET and OUTPUT_ROOT must be different.")

    if MAX_IMAGES_PER_SPLIT is not None and MAX_IMAGES_PER_SPLIT <= 0:
        raise ValueError("MAX_IMAGES_PER_SPLIT must be positive or None.")

    for config in GAUSSIAN_CONFIGS:
        kernel = config["kernel_size"]
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(
                f"Gaussian kernel must be positive and odd: {config}"
            )
        if config["sigma"] <= 0:
            raise ValueError(f"Gaussian sigma must be positive: {config}")

    for config in LBP_CONFIGS:
        if config["points"] <= 0 or config["radius"] <= 0:
            raise ValueError(f"LBP P and R must be positive: {config}")

    for split in SPLITS:
        images = SOURCE_DATASET / split / "images"
        labels = SOURCE_DATASET / split / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(
                f"Expected '{split}/images' and '{split}/labels' under "
                f"{SOURCE_DATASET}"
            )


def process_split(split, output_dataset, gaussian_config, lbp_config):
    source_images = SOURCE_DATASET / split / "images"
    source_labels = SOURCE_DATASET / split / "labels"
    output_images = output_dataset / split / "images"
    output_labels = output_dataset / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(source_images)
    if not image_files:
        raise RuntimeError(f"No supported images found in {source_images}")

    processed_count = 0
    missing_labels = []

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")

        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            if STRICT_LABELS:
                raise FileNotFoundError(
                    f"Missing YOLO label for {image_path.name}: {label_path}"
                )
            print(f"WARNING: skipping image without label: {image_path.name}")
            continue

        processed = preprocess_image(
            image,
            gaussian_config,
            lbp_config,
        )

        # PNG avoids introducing JPEG texture artefacts into the LBP image.
        output_image = output_images / f"{image_path.stem}.png"
        if not cv2.imwrite(str(output_image), processed):
            raise RuntimeError(f"Could not write {output_image}")

        shutil.copy2(label_path, output_labels / f"{image_path.stem}.txt")
        processed_count += 1

        if index % 200 == 0 or index == len(image_files):
            print(f"  {split}: {index}/{len(image_files)}")

    return {
        "split": split,
        "processed_images": processed_count,
        "missing_labels": len(missing_labels),
    }


def create_yaml(output_dataset):
    names = "\n".join(
        f"  {class_id}: {class_name}"
        for class_id, class_name in CLASS_NAMES.items()
    )

    content = (
        f'path: "{output_dataset.resolve().as_posix()}"\n\n'
        "train: train/images\n"
        "val: val/images\n\n"
        "names:\n"
        f"{names}\n"
    )

    (output_dataset / "data.yaml").write_text(content, encoding="utf-8")


def save_manifest(output_dataset, name, gaussian_config, lbp_config, counts):
    manifest = {
        "configuration": name,
        "source_dataset": str(SOURCE_DATASET.resolve()),
        "output_dataset": str(output_dataset.resolve()),
        "splits": list(SPLITS),
        "max_images_per_split": MAX_IMAGES_PER_SPLIT,
        "output_format": "PNG",
        "output_channels_bgr": ["gaussian", "lbp", "gaussian"],
        "gaussian": gaussian_config,
        "lbp": {**lbp_config, "method": LBP_METHOD},
        "counts": counts,
    }

    path = output_dataset / "preprocessing.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_dataset(gaussian_config, lbp_config):
    name = configuration_name(gaussian_config, lbp_config)
    output_dataset = OUTPUT_ROOT / name

    print("\n" + "=" * 80)
    print(f"GENERATING {name}")
    print("=" * 80)

    counts = []
    for split in SPLITS:
        counts.append(
            process_split(
                split,
                output_dataset,
                gaussian_config,
                lbp_config,
            )
        )

    create_yaml(output_dataset)
    save_manifest(
        output_dataset,
        name,
        gaussian_config,
        lbp_config,
        counts,
    )

    return {
        "configuration": name,
        "kernel_size": gaussian_config["kernel_size"],
        "sigma": gaussian_config["sigma"],
        "lbp_points": lbp_config["points"],
        "lbp_radius": lbp_config["radius"],
        "lbp_method": LBP_METHOD,
        "dataset_path": str(output_dataset.resolve()),
    }


def save_parameter_grid(rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_ROOT / "parameter_grid.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nParameter grid saved to: {output_file}")


def main():
    validate_settings()

    print("=" * 80)
    print("GAUSSIAN + LBP DATASET GENERATOR")
    print("=" * 80)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Output: {OUTPUT_ROOT}")
    print(
        f"Configurations: {len(GAUSSIAN_CONFIGS) * len(LBP_CONFIGS)}"
    )

    rows = []
    for gaussian_config, lbp_config in product(
        GAUSSIAN_CONFIGS,
        LBP_CONFIGS,
    ):
        rows.append(create_dataset(gaussian_config, lbp_config))
        save_parameter_grid(rows)

    print("\n" + "=" * 80)
    print("ALL PARAMETER DATASETS GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
