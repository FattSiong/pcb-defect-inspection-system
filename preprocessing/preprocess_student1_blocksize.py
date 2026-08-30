from pathlib import Path
import cv2
import shutil


# =========================================================
# SETTINGS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_student1_blocksize"
)


# =========================================================
# FIXED STUDENT 1 PARAMETERS
# =========================================================

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ADAPTIVE_C = 2

# Only blockSize changes
BLOCK_SIZES = [
    11,
    21,
    31,
    41,
    51
]


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}


# =========================================================
# PREPROCESS ONE IMAGE
# =========================================================

def preprocess_image(image, block_size):

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(gray)

    # -----------------------------------------------------
    # Adaptive Gaussian Thresholding
    # -----------------------------------------------------

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        ADAPTIVE_C
    )

    return thresholded


# =========================================================
# GET IMAGE FILES
# =========================================================

def get_image_files(folder):

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    return sorted(
        [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in supported_extensions
            )
        ]
    )


# =========================================================
# PROCESS ONE SPLIT
# =========================================================

def process_split(
    split,
    output_dataset,
    block_size
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


    print(
        f"\nProcessing {split} "
        f"with blockSize={block_size}"
    )

    print(
        f"Images found: {len(image_files)}"
    )


    processed_count = 0
    missing_labels = 0


    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # -------------------------------------------------
        # Read original image
        # -------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Cannot read "
                f"{image_path.name}"
            )

            continue


        # -------------------------------------------------
        # Preprocess image
        # -------------------------------------------------

        processed = preprocess_image(
            image,
            block_size
        )


        # -------------------------------------------------
        # Save processed image
        # -------------------------------------------------

        output_image = (
            output_images /
            image_path.name
        )

        cv2.imwrite(
            str(output_image),
            processed
        )


        # -------------------------------------------------
        # Copy corresponding YOLO label
        # -------------------------------------------------

        label_path = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        output_label = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        if label_path.exists():

            shutil.copy2(
                label_path,
                output_label
            )

        else:

            missing_labels += 1

            print(
                f"WARNING: Missing label for "
                f"{image_path.name}"
            )


        processed_count += 1


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            index % 200 == 0
            or index == len(image_files)
        ):

            print(
                f"{split}: "
                f"{index}/{len(image_files)}"
            )


    print(
        f"\nProcessed images: "
        f"{processed_count}"
    )

    print(
        f"Missing labels: "
        f"{missing_labels}"
    )


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_yaml(output_dataset):

    yaml_path = (
        output_dataset /
        "data.yaml"
    )


    content = f"""path: {output_dataset.as_posix()}

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

        file.write(content)


    print(
        f"data.yaml created:\n"
        f"{yaml_path}"
    )


# =========================================================
# VERIFY DATASET
# =========================================================

def verify_dataset(
    output_dataset,
    block_size
):

    print("\n" + "-" * 70)
    print(
        f"VERIFYING BLOCK SIZE {block_size}"
    )
    print("-" * 70)


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
            label_folder.glob(
                "*.txt"
            )
        )


        print(
            f"{split:<6} "
            f"images = {len(images)}, "
            f"labels = {len(labels)}"
        )


# =========================================================
# CREATE ONE BLOCKSIZE DATASET
# =========================================================

def create_dataset(block_size):

    output_dataset = (
        OUTPUT_ROOT /
        f"block_{block_size}"
    )


    print("\n" + "=" * 80)
    print(
        f"CREATING BLOCK SIZE {block_size} DATASET"
    )
    print("=" * 80)


    # -----------------------------------------------------
    # Process training images
    # -----------------------------------------------------

    process_split(
        "train",
        output_dataset,
        block_size
    )


    # -----------------------------------------------------
    # Process validation images
    # -----------------------------------------------------

    process_split(
        "val",
        output_dataset,
        block_size
    )


    # -----------------------------------------------------
    # Create YAML
    # -----------------------------------------------------

    create_yaml(
        output_dataset
    )


    # -----------------------------------------------------
    # Verify result
    # -----------------------------------------------------

    verify_dataset(
        output_dataset,
        block_size
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 BLOCK SIZE PREPROCESSING")
    print("=" * 80)

    print(
        f"Source dataset:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        f"\nCLAHE clipLimit = "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tileGridSize = "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Adaptive C = "
        f"{ADAPTIVE_C}"
    )

    print(
        f"Block sizes = "
        f"{BLOCK_SIZES}"
    )


    for block_size in BLOCK_SIZES:

        create_dataset(
            block_size
        )


    print("\n" + "=" * 80)
    print("ALL PREPROCESSING COMPLETED")
    print("=" * 80)

    print(
        "\nProcessed datasets saved to:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()