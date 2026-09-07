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
    r"C:\Users\user\Downloads\pcb_dataset_student2_final_test"
)


# =========================================================
# FINAL WINNING PARAMETERS (PHASE 4 WINNER)
# =========================================================

CANNY_THRESH = (50, 150)
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 5

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180


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

def preprocess_image(image):

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
# PROCESS SPLIT (TEST ONLY)
# =========================================================

def process_test_split():

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
        OUTPUT_ROOT /
        "test" /
        "images"
    )

    output_labels = (
        OUTPUT_ROOT /
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

    processed_count = 0
    missing_labels = 0

    print("\n" + "-" * 80)
    print("PROCESSING TEST SPLIT WITH FINAL WINNING CONFIGURATION")
    print("-" * 80)

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(f"WARNING: Failed to read {image_path.name}")
            continue

        processed = preprocess_image(image)

        output_image_path = (
            output_images /
            image_path.name
        )

        if not cv2.imwrite(
            str(output_image_path),
            processed
        ):
            print(f"WARNING: Failed to save {image_path.name}")
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
            print(f"WARNING: Missing label for {image_path.name}")

        processed_count += 1

        if index % 200 == 0 or index == len(image_files):
            print(f"{index}/{len(image_files)}")

    print(f"\nProcessed      : {processed_count}")
    print(f"Missing labels : {missing_labels}")


# =========================================================
# CREATE DATA.YAML (POINTS TO TEST SPLIT)
# =========================================================

def create_yaml():

    yaml_path = (
        OUTPUT_ROOT /
        "data.yaml"
    )

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}

train: test/images
val: test/images
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
        file.write(yaml_content)

    print(f"Created:\n{yaml_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 80)
    print("STUDENT 2 - FINAL TEST DATASET GENERATION")
    print("=" * 80)

    process_test_split()
    create_yaml()

    print("\n" + "=" * 80)
    print("FINAL TEST PREPROCESSING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()