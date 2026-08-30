from pathlib import Path
import cv2
import shutil


# =========================================================
# PATH SETTINGS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_student1_c"
)


# =========================================================
# FIXED PARAMETERS
# =========================================================

# CLAHE remains fixed
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Best block size from previous experiment
BLOCK_SIZE = 41

# C = 2 already exists from block-size experiment.
# Only create NEW datasets for C = 5 and C = 8.
C_VALUES = [
    5,
    8
]


# =========================================================
# SUPPORTED IMAGE EXTENSIONS
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# GET IMAGE FILES
# =========================================================

def get_image_files(folder):

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder not found:\n{folder}"
        )

    return sorted(
        [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


# =========================================================
# PREPROCESS ONE IMAGE
# =========================================================

def preprocess_image(image, c_value):

    # -----------------------------------------------------
    # Step 1: Convert to grayscale
    # -----------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image


    # -----------------------------------------------------
    # Step 2: Apply CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(
        gray
    )


    # -----------------------------------------------------
    # Step 3: Adaptive Gaussian Thresholding
    #
    # ONLY C changes in this experiment.
    # -----------------------------------------------------

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        c_value
    )

    return thresholded


# =========================================================
# PROCESS ONE DATASET SPLIT
# =========================================================

def process_split(
    split,
    output_dataset,
    c_value
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


    print("\n" + "-" * 80)

    print(
        f"Processing {split.upper()} "
        f"with C = {c_value}"
    )

    print("-" * 80)

    print(
        f"Images found: {len(image_files)}"
    )


    processed_count = 0
    missing_labels = 0
    failed_images = 0


    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # -------------------------------------------------
        # Read source image
        # -------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            failed_images += 1

            print(
                f"WARNING: Could not read "
                f"{image_path.name}"
            )

            continue


        # -------------------------------------------------
        # Apply CLAHE + Adaptive Gaussian Thresholding
        # -------------------------------------------------

        processed = preprocess_image(
            image,
            c_value
        )


        # -------------------------------------------------
        # Save processed image
        # -------------------------------------------------

        output_image_path = (
            output_images /
            image_path.name
        )

        success = cv2.imwrite(
            str(output_image_path),
            processed
        )

        if not success:

            failed_images += 1

            print(
                f"WARNING: Could not save "
                f"{output_image_path}"
            )

            continue


        # -------------------------------------------------
        # Copy YOLO label
        # -------------------------------------------------

        source_label_path = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        output_label_path = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        if source_label_path.exists():

            shutil.copy2(
                source_label_path,
                output_label_path
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
        f"\nProcessed images : {processed_count}"
    )

    print(
        f"Missing labels   : {missing_labels}"
    )

    print(
        f"Failed images    : {failed_images}"
    )


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_yaml(output_dataset):

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
        f"\ndata.yaml created:\n"
        f"{yaml_path}"
    )


# =========================================================
# VERIFY OUTPUT DATASET
# =========================================================

def verify_dataset(
    output_dataset,
    c_value
):

    print("\n" + "=" * 80)

    print(
        f"VERIFYING DATASET FOR C = {c_value}"
    )

    print("=" * 80)


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


        # -------------------------------------------------
        # Check that every image has a matching label
        # -------------------------------------------------

        unmatched_images = []

        for image_path in images:

            label_path = (
                label_folder /
                f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                unmatched_images.append(
                    image_path.name
                )


        print(
            f"{split:<6} "
            f"images = {len(images)}, "
            f"labels = {len(labels)}, "
            f"unmatched = {len(unmatched_images)}"
        )


        if unmatched_images:

            print(
                "WARNING: Some images have no "
                "matching label."
            )


# =========================================================
# CREATE ONE C DATASET
# =========================================================

def create_dataset(c_value):

    output_dataset = (
        OUTPUT_ROOT /
        f"c_{c_value}"
    )


    print("\n" + "=" * 80)

    print(
        f"CREATING DATASET FOR C = {c_value}"
    )

    print("=" * 80)


    print(
        f"CLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tile grid : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Block Size      : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C               : "
        f"{c_value}"
    )


    # -----------------------------------------------------
    # Process TRAIN
    # -----------------------------------------------------

    process_split(
        "train",
        output_dataset,
        c_value
    )


    # -----------------------------------------------------
    # Process VALIDATION
    # -----------------------------------------------------

    process_split(
        "val",
        output_dataset,
        c_value
    )


    # -----------------------------------------------------
    # Create data.yaml
    # -----------------------------------------------------

    create_yaml(
        output_dataset
    )


    # -----------------------------------------------------
    # Verify output
    # -----------------------------------------------------

    verify_dataset(
        output_dataset,
        c_value
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 - C PARAMETER PREPROCESSING")
    print("=" * 80)


    print(
        f"\nSource dataset:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_ROOT}"
    )


    print(
        f"\nFixed CLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"Fixed tileGridSize    : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Fixed blockSize       : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C values to create    : "
        f"{C_VALUES}"
    )


    for c_value in C_VALUES:

        create_dataset(
            c_value
        )


    print("\n" + "=" * 80)
    print("C PARAMETER PREPROCESSING COMPLETED")
    print("=" * 80)


    print(
        "\nCreated only:"
    )

    for c_value in C_VALUES:

        print(
            OUTPUT_ROOT /
            f"c_{c_value}"
        )


if __name__ == "__main__":
    main()