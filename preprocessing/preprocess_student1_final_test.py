from pathlib import Path
import cv2
import shutil
import csv
import time


# =========================================================
# PATHS
# =========================================================

# Original reduced dataset
SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

# Final processed TEST dataset
OUTPUT_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_student1_final_test"
)

# Existing processed TRAIN/VAL dataset for the winning clipLimit = 1.0
# These are referenced in data.yaml only.
EXISTING_FINAL_TRAIN_VAL = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_cliplimit"
    r"\clip_1.0"
)

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

SUMMARY_DIR = (
    PROJECT_ROOT /
    "runs" /
    "student1_final_test"
)


# =========================================================
# FINAL LOCKED PARAMETERS
# =========================================================

CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = (8, 8)

BLOCK_SIZE = 41
ADAPTIVE_C = 2


# =========================================================
# IMAGE SETTINGS
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
            f"Folder not found:\n{folder}"
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
# FINAL PREPROCESSING PIPELINE
# =========================================================

def preprocess_image(image):

    # -----------------------------------------------------
    # Step 1: Grayscale
    # -----------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image


    # -----------------------------------------------------
    # Step 2: CLAHE
    # FINAL clipLimit = 1.0
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
    # FINAL blockSize = 41
    # FINAL C = 2
    # -----------------------------------------------------

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
# PROCESS TEST SET
# =========================================================

def process_test_set():

    source_images = (
        SOURCE_DATASET /
        "test" /
        "images"
    )

    source_labels = (
        SOURCE_DATASET /
        "test" /
        "labels"
    )

    output_images = (
        OUTPUT_DATASET /
        "test" /
        "images"
    )

    output_labels = (
        OUTPUT_DATASET /
        "test" /
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


    print("\n" + "=" * 80)
    print("PROCESSING FINAL TEST DATASET")
    print("=" * 80)

    print(
        f"Test images found: "
        f"{len(image_files)}"
    )


    processed_count = 0
    missing_labels = 0
    failed_images = 0

    individual_times_ms = []


    total_start = time.perf_counter()


    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # -------------------------------------------------
        # Read image
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
        # Time preprocessing
        # -------------------------------------------------

        start = time.perf_counter()

        processed = preprocess_image(
            image
        )

        end = time.perf_counter()


        preprocessing_ms = (
            end - start
        ) * 1000

        individual_times_ms.append(
            preprocessing_ms
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
                f"WARNING: Failed to save "
                f"{output_image_path}"
            )

            continue


        # -------------------------------------------------
        # Copy YOLO label
        # -------------------------------------------------

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
                f"WARNING: Missing label for "
                f"{image_path.name}"
            )


        processed_count += 1


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            index % 100 == 0
            or index == len(image_files)
        ):

            print(
                f"{index}/"
                f"{len(image_files)}"
            )


    total_end = time.perf_counter()

    total_seconds = (
        total_end - total_start
    )


    if individual_times_ms:

        average_preprocessing_ms = (
            sum(individual_times_ms)
            / len(individual_times_ms)
        )

    else:

        average_preprocessing_ms = 0.0


    print("\n" + "-" * 80)

    print(
        f"Successfully processed : "
        f"{processed_count}"
    )

    print(
        f"Missing labels         : "
        f"{missing_labels}"
    )

    print(
        f"Failed images          : "
        f"{failed_images}"
    )

    print(
        f"Total elapsed time     : "
        f"{total_seconds:.3f} seconds"
    )

    print(
        f"Average preprocessing  : "
        f"{average_preprocessing_ms:.3f} ms/image"
    )


    return {
        "Total Test Images":
            len(image_files),

        "Successfully Processed":
            processed_count,

        "Missing Labels":
            missing_labels,

        "Failed Images":
            failed_images,

        "Total Preprocessing Time (seconds)":
            total_seconds,

        "Average Preprocessing Time (ms/image)":
            average_preprocessing_ms
    }


# =========================================================
# CREATE FINAL DATA.YAML
# =========================================================

def create_yaml():

    yaml_path = (
        OUTPUT_DATASET /
        "data.yaml"
    )


    # Train and val reference the winning clipLimit=1.0 dataset.
    # Test points to the newly processed final test images.
    #
    # The test script will explicitly use split="test".

    train_path = (
        EXISTING_FINAL_TRAIN_VAL /
        "train" /
        "images"
    ).as_posix()

    val_path = (
        EXISTING_FINAL_TRAIN_VAL /
        "val" /
        "images"
    ).as_posix()


    yaml_content = f"""path: {OUTPUT_DATASET.as_posix()}

train: {train_path}
val: {val_path}
test: test/images

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
        f"\nFinal data.yaml created:\n"
        f"{yaml_path}"
    )

    return yaml_path


# =========================================================
# VERIFY FINAL TEST DATASET
# =========================================================

def verify_test_dataset():

    image_folder = (
        OUTPUT_DATASET /
        "test" /
        "images"
    )

    label_folder = (
        OUTPUT_DATASET /
        "test" /
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


    unmatched = []


    for image_path in images:

        expected_label = (
            label_folder /
            f"{image_path.stem}.txt"
        )

        if not expected_label.exists():

            unmatched.append(
                image_path.name
            )


    print("\n" + "=" * 80)
    print("FINAL TEST DATASET VERIFICATION")
    print("=" * 80)

    print(
        f"Images    : "
        f"{len(images)}"
    )

    print(
        f"Labels    : "
        f"{len(labels)}"
    )

    print(
        f"Unmatched : "
        f"{len(unmatched)}"
    )


    if unmatched:

        print(
            "\nWARNING: Some test images "
            "do not have matching labels."
        )


# =========================================================
# SAVE PREPROCESSING SUMMARY
# =========================================================

def save_summary(summary):

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    summary_path = (
        SUMMARY_DIR /
        "final_test_preprocessing_summary.csv"
    )


    summary.update({
        "CLAHE Clip Limit":
            CLAHE_CLIP_LIMIT,

        "CLAHE Tile Grid":
            "8x8",

        "Adaptive Block Size":
            BLOCK_SIZE,

        "Adaptive C":
            ADAPTIVE_C
    })


    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary.keys()
            )
        )

        writer.writeheader()

        writer.writerow(
            summary
        )


    print(
        f"\nPreprocessing summary saved:\n"
        f"{summary_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 - FINAL TECHNIQUE A TEST PREPROCESSING")
    print("=" * 80)


    print(
        "\nFINAL LOCKED PARAMETERS"
    )

    print(
        f"CLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tileGrid  : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"blockSize       : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C               : "
        f"{ADAPTIVE_C}"
    )


    # -----------------------------------------------------
    # Make sure the winning train/val dataset exists
    # -----------------------------------------------------

    if not EXISTING_FINAL_TRAIN_VAL.exists():

        raise FileNotFoundError(
            f"Winning clipLimit=1.0 dataset "
            f"not found:\n"
            f"{EXISTING_FINAL_TRAIN_VAL}"
        )


    # -----------------------------------------------------
    # Process TEST only
    # -----------------------------------------------------

    summary = process_test_set()


    # -----------------------------------------------------
    # Create data.yaml
    # -----------------------------------------------------

    create_yaml()


    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    verify_test_dataset()


    # -----------------------------------------------------
    # Save timing / processing information
    # -----------------------------------------------------

    save_summary(
        summary
    )


    print("\n" + "=" * 80)
    print("FINAL TEST PREPROCESSING COMPLETED")
    print("=" * 80)

    print(
        f"\nProcessed test dataset:\n"
        f"{OUTPUT_DATASET}"
    )


if __name__ == "__main__":
    main()