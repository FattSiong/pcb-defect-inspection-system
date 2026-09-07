from pathlib import Path
import cv2
import numpy as np
import shutil


# =========================================================
# PATHS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\user\Downloads\pcb_dataset_reduced"
)

OUTPUT_ROOT = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_max_gap"
)


# =========================================================
# FIXED PARAMETERS
# =========================================================

CANNY_THRESH = (50, 150)
HOUGH_THRESHOLD = 80

# Updated to reflect the winning Phase 3 configuration
HOUGH_MIN_LINE_LENGTH = 60

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180

# maxLineGap = 10 already exists from Phase 3
# Only generate NEW datasets
MAX_LINE_GAPS = [
    2,
    5
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
    max_gap
):


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

    # Hough Line Transform (Varying maxLineGap)
    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=max_gap
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
# PROCESS SPLIT
# =========================================================

def process_split(
    split,
    output_dataset,
    max_gap
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
        f"maxLineGap = {max_gap}"
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
            max_gap
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
    max_gap
):

    folder_name = (
        f"max_gap_{max_gap}"
    )

    output_dataset = (
        OUTPUT_ROOT /
        folder_name
    )


    print("\n" + "=" * 80)

    print(
        f"CREATING maxLineGap = "
        f"{max_gap}"
    )

    print("=" * 80)

    print(
        f"Canny      = "
        f"{CANNY_THRESH}"
    )

    print(
        f"Hough Thr  = "
        f"{HOUGH_THRESHOLD}"
    )

    print(
        f"Min Length = "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )



    for split in [
        "train",
        "val"
    ]:

        process_split(
            split,
            output_dataset,
            max_gap
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
    print("STUDENT 2 - MAX LINE GAP PREPROCESSING (PHASE 4)")
    print("=" * 80)

    print(
        f"\nSource:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nFixed Canny Threshold   : "
        f"{CANNY_THRESH}"
    )

    print(
        f"Fixed Hough Threshold   : "
        f"{HOUGH_THRESHOLD}"
    )

    print(
        f"Fixed Min Line Length   : "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"New maxLineGaps         : "
        f"{MAX_LINE_GAPS}"
    )


    for max_gap in MAX_LINE_GAPS:

        create_dataset(
            max_gap
        )


    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()