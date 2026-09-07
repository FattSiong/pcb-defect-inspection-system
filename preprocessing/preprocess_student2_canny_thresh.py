from pathlib import Path
import cv2
import numpy as np
import shutil


# =========================================================
# SETTINGS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\user\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\user\Downloads\pcb_dataset_student2_canny_thresh"
)


# =========================================================
# FIXED STUDENT 2 PARAMETERS (HOUGH BASELINES)
# =========================================================

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 10

# Only Canny thresholds change in Phase 1
CANNY_THRESHOLDS = [
    (50, 150),
    (100, 200),
    (150, 250)
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

def preprocess_image(image, canny_thresh):

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # Canny Edge Detection
    # -----------------------------------------------------
    
    lower_thresh, upper_thresh = canny_thresh

    edges = cv2.Canny(
        blurred,
        lower_thresh,
        upper_thresh
    )

    # -----------------------------------------------------
    # Hough Line Transform
    # -----------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
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
    canny_thresh
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
        f"with Canny={canny_thresh}"
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
            canny_thresh
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
    canny_thresh
):

    print("\n" + "-" * 70)
    print(
        f"VERIFYING CANNY {canny_thresh}"
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
# CREATE ONE CONFIGURATION DATASET
# =========================================================

def create_dataset(canny_thresh):

    output_dataset = (
        OUTPUT_ROOT /
        f"canny_{canny_thresh[0]}_{canny_thresh[1]}"
    )


    print("\n" + "=" * 80)
    print(
        f"CREATING CANNY {canny_thresh} DATASET"
    )
    print("=" * 80)


    # -----------------------------------------------------
    # Process training images
    # -----------------------------------------------------

    process_split(
        "train",
        output_dataset,
        canny_thresh
    )


    # -----------------------------------------------------
    # Process validation images
    # -----------------------------------------------------

    process_split(
        "val",
        output_dataset,
        canny_thresh
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
        canny_thresh
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 2 CANNY THRESHOLD PREPROCESSING (PHASE 1)")
    print("=" * 80)

    print(
        f"Source dataset:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nOutput root:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        f"\nFixed Hough Threshold = "
        f"{HOUGH_THRESHOLD}"
    )

    print(
        f"Fixed Min Line Length = "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"Fixed Max Line Gap = "
        f"{HOUGH_MAX_LINE_GAP}"
    )

    print(
        f"Canny Thresholds = "
        f"{CANNY_THRESHOLDS}"
    )


    for canny_thresh in CANNY_THRESHOLDS:

        create_dataset(
            canny_thresh
        )


    print("\n" + "=" * 80)
    print("PHASE 1 PREPROCESSING COMPLETED")
    print("=" * 80)

    print(
        "\nProcessed datasets saved to:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()