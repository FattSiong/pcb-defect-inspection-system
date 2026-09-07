from pathlib import Path
import cv2
import numpy as np
import shutil


# =========================================================
# PATH SETTINGS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\user\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\user\Downloads\pcb_dataset_student2_hough_thresh"
)


# =========================================================
# FIXED PARAMETERS
# =========================================================

# Best Canny Threshold from Phase 1 experiment
CANNY_THRESH = (50, 150)

# Other Hough parameters remain fixed
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 10

# Hough Threshold = 50 already exists from Phase 1 (canny_50_150).
# Only create NEW datasets for 30 and 80.
HOUGH_THRESHOLDS = [
    30,
    80
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

def preprocess_image(image, hough_thresh):

    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    
    edges = cv2.Canny(
        blurred,
        CANNY_THRESH[0],
        CANNY_THRESH[1]
    )


    
    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        hough_thresh,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )


    output_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                output_bgr,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return output_bgr


# =========================================================
# PROCESS ONE DATASET SPLIT
# =========================================================

def process_split(
    split,
    output_dataset,
    hough_thresh
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
        f"with Hough Threshold = {hough_thresh}"
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
        # Apply Canny + Hough
        # -------------------------------------------------

        processed = preprocess_image(
            image,
            hough_thresh
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
    hough_thresh
):

    print("\n" + "=" * 80)

    print(
        f"VERIFYING DATASET FOR HOUGH THRESHOLD = {hough_thresh}"
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
# CREATE ONE HOUGH DATASET
# =========================================================

def create_dataset(hough_thresh):

    output_dataset = (
        OUTPUT_ROOT /
        f"hough_{hough_thresh}"
    )


    print("\n" + "=" * 80)

    print(
        f"CREATING DATASET FOR HOUGH THRESHOLD = {hough_thresh}"
    )

    print("=" * 80)


    print(
        f"Fixed Canny     : "
        f"{CANNY_THRESH}"
    )

    print(
        f"Fixed Min Length: "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"Fixed Max Gap   : "
        f"{HOUGH_MAX_LINE_GAP}"
    )

    print(
        f"Hough Threshold : "
        f"{hough_thresh}"
    )


    # -----------------------------------------------------
    # Process TRAIN
    # -----------------------------------------------------

    process_split(
        "train",
        output_dataset,
        hough_thresh
    )


    # -----------------------------------------------------
    # Process VALIDATION
    # -----------------------------------------------------

    process_split(
        "val",
        output_dataset,
        hough_thresh
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
        hough_thresh
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 2 - HOUGH THRESHOLD PREPROCESSING (PHASE 2)")
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
        f"\nFixed Canny          : "
        f"{CANNY_THRESH}"
    )

    print(
        f"Fixed Min Line Length: "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"Fixed Max Line Gap   : "
        f"{HOUGH_MAX_LINE_GAP}"
    )

    print(
        f"Hough values to create: "
        f"{HOUGH_THRESHOLDS}"
    )


    for hough_thresh in HOUGH_THRESHOLDS:

        create_dataset(
            hough_thresh
        )


    print("\n" + "=" * 80)
    print("HOUGH THRESHOLD PREPROCESSING COMPLETED")
    print("=" * 80)


    print(
        "\nCreated only:"
    )

    for hough_thresh in HOUGH_THRESHOLDS:

        print(
            OUTPUT_ROOT /
            f"hough_{hough_thresh}"
        )


if __name__ == "__main__":
    main()