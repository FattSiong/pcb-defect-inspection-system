from pathlib import Path
import cv2
import shutil


# =========================================================
# PATHS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_cliplimit"
)


# =========================================================
# FIXED PARAMETERS
# =========================================================

CLAHE_TILE_GRID_SIZE = (
    8,
    8
)

BLOCK_SIZE = 41
ADAPTIVE_C = 2

# clipLimit = 2.0 already exists
# Only generate NEW datasets
CLIP_LIMITS = [
    1.0,
    3.0
]


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# GET IMAGES
# =========================================================

def get_image_files(folder):

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n"
            f"{folder}"
        )

    return sorted([
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(
    image,
    clip_limit
):

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(
        gray
    )

    # Adaptive Gaussian Thresholding
    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        ADAPTIVE_C
    )

    return thresholded


# =========================================================
# PROCESS SPLIT
# =========================================================

def process_split(
    split,
    output_dataset,
    clip_limit
):

    source_images = (
        SOURCE_DATASET /
        split /
        "images"
    )

    source_labels = (
        SOURCE_DATASET /
        split /
        "labels"
    )

    output_images = (
        output_dataset /
        split /
        "images"
    )

    output_labels = (
        output_dataset /
        split /
        "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = get_image_files(
        source_images
    )

    processed_count = 0
    missing_labels = 0


    print("\n" + "-" * 80)

    print(
        f"{split.upper()} | "
        f"clipLimit = {clip_limit}"
    )

    print("-" * 80)


    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Failed to read "
                f"{image_path.name}"
            )

            continue


        processed = preprocess_image(
            image,
            clip_limit
        )


        output_image_path = (
            output_images /
            image_path.name
        )

        if not cv2.imwrite(
            str(output_image_path),
            processed
        ):

            print(
                f"WARNING: Failed to save "
                f"{image_path.name}"
            )

            continue


        source_label = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        output_label = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        if source_label.exists():

            shutil.copy2(
                source_label,
                output_label
            )

        else:

            missing_labels += 1

            print(
                f"WARNING: Missing label "
                f"for {image_path.name}"
            )


        processed_count += 1


        if (
            index % 200 == 0
            or index == len(image_files)
        ):

            print(
                f"{index}/"
                f"{len(image_files)}"
            )


    print(
        f"Processed      : "
        f"{processed_count}"
    )

    print(
        f"Missing labels : "
        f"{missing_labels}"
    )


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_yaml(
    output_dataset
):

    yaml_path = (
        output_dataset /
        "data.yaml"
    )

    yaml_content = f"""path: {output_dataset.as_posix()}

train: train/images
val: val/images

names:
  0: mouse_bite
  1: spur
  2: missing_hole
  3: short
  4: open_circuit
  5: spurious_copper
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            yaml_content
        )

    print(
        f"Created:\n"
        f"{yaml_path}"
    )


# =========================================================
# VERIFY
# =========================================================

def verify_dataset(
    output_dataset
):

    print("\nVerification:")

    for split in [
        "train",
        "val"
    ]:

        image_folder = (
            output_dataset /
            split /
            "images"
        )

        label_folder = (
            output_dataset /
            split /
            "labels"
        )

        images = get_image_files(
            image_folder
        )

        labels = list(
            label_folder.glob("*.txt")
        )

        missing = []

        for image_path in images:

            label = (
                label_folder /
                f"{image_path.stem}.txt"
            )

            if not label.exists():

                missing.append(
                    image_path.name
                )

        print(
            f"{split:<6} | "
            f"images={len(images)} | "
            f"labels={len(labels)} | "
            f"unmatched={len(missing)}"
        )


# =========================================================
# CREATE ONE DATASET
# =========================================================

def create_dataset(
    clip_limit
):

    folder_name = (
        f"clip_{clip_limit:.1f}"
    )

    output_dataset = (
        OUTPUT_ROOT /
        folder_name
    )


    print("\n" + "=" * 80)

    print(
        f"CREATING clipLimit = "
        f"{clip_limit}"
    )

    print("=" * 80)

    print(
        f"blockSize = "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C         = "
        f"{ADAPTIVE_C}"
    )

    print(
        f"tileGrid  = "
        f"{CLAHE_TILE_GRID_SIZE}"
    )


    # Parameter tuning uses train + val only
    for split in [
        "train",
        "val"
    ]:

        process_split(
            split,
            output_dataset,
            clip_limit
        )


    create_yaml(
        output_dataset
    )

    verify_dataset(
        output_dataset
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 - CLAHE CLIPLIMIT PREPROCESSING")
    print("=" * 80)

    print(
        f"\nSource:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nFixed blockSize : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Fixed C         : "
        f"{ADAPTIVE_C}"
    )

    print(
        f"New clipLimits  : "
        f"{CLIP_LIMITS}"
    )


    for clip_limit in CLIP_LIMITS:

        create_dataset(
            clip_limit
        )


    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()