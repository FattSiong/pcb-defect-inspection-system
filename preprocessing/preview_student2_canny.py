from pathlib import Path
import cv2
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\user\Downloads\pcb_dataset_reduced"
)

SOURCE_IMAGES = DATASET_ROOT / "train" / "images"
SOURCE_LABELS = DATASET_ROOT / "train" / "labels"

OUTPUT_DIR = Path(
    r"D:\pcb-defect-inspection-system"
    r"\runs\student2_preview"
)

# ---------------------------------------------------------
# Student 2 fixed Hough Transform settings
# ---------------------------------------------------------

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# ---------------------------------------------------------
# Canny Thresholds currently being investigated
# (Lower, Upper)
# ---------------------------------------------------------

CANNY_THRESHOLDS = [
    (50, 150),
    (100, 200),
    (150, 250)
]


# =========================================================
# PCB DEFECT CLASSES
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
# READ YOLO LABEL
# =========================================================

def read_yolo_labels(label_path):

    labels = []

    if not label_path.exists():
        return labels

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    return labels


# =========================================================
# FIND ONE IMAGE FOR EACH DEFECT CLASS
# =========================================================

def find_sample_for_class(target_class):

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]

    image_files = []

    for extension in image_extensions:

        image_files.extend(
            SOURCE_IMAGES.glob(extension)
        )

    image_files = sorted(image_files)


    for image_path in image_files:

        label_path = (
            SOURCE_LABELS /
            f"{image_path.stem}.txt"
        )

        labels = read_yolo_labels(
            label_path
        )

        if not labels:
            continue

        unique_classes = {
            label["class_id"]
            for label in labels
        }

        if (
            target_class in unique_classes
            and len(unique_classes) == 1
        ):

            return image_path, labels


    for image_path in image_files:

        label_path = (
            SOURCE_LABELS /
            f"{image_path.stem}.txt"
        )

        labels = read_yolo_labels(
            label_path
        )

        unique_classes = {
            label["class_id"]
            for label in labels
        }

        if target_class in unique_classes:

            return image_path, labels

    return None, None


# =========================================================
# APPLY CANNY EDGE DETECTION + HOUGH TRANSFORM
# =========================================================

def preprocess_image(
    image,
    canny_thresh
):

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # Apply Gaussian Blur to reduce noise
    # -----------------------------------------------------
    
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # Apply Canny Edge Detection
    # -----------------------------------------------------
    
    lower_thresh, upper_thresh = canny_thresh

    edges = cv2.Canny(
        blurred,
        lower_thresh,
        upper_thresh
    )

    # -----------------------------------------------------
    # Apply Hough Line Transform
    # -----------------------------------------------------
    
    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    # Convert binary edge map to BGR so we can draw 
    # colored Hough lines and YOLO bounding boxes
    output_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    # Draw detected Hough lines in green
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
# DRAW YOLO BOUNDING BOXES
# =========================================================

def draw_labels(
    image,
    labels,
    target_class=None
):

    output = image.copy()

    image_height, image_width = (
        output.shape[:2]
    )

    for label in labels:

        class_id = label["class_id"]

        # Only draw the requested defect class
        # when target_class is specified
        if (
            target_class is not None
            and class_id != target_class
        ):
            continue

        x_center = (
            label["x_center"]
            * image_width
        )

        y_center = (
            label["y_center"]
            * image_height
        )

        box_width = (
            label["width"]
            * image_width
        )

        box_height = (
            label["height"]
            * image_height
        )

        x1 = int(
            x_center
            - box_width / 2
        )

        y1 = int(
            y_center
            - box_height / 2
        )

        x2 = int(
            x_center
            + box_width / 2
        )

        y2 = int(
            y_center
            + box_height / 2
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            image_width - 1,
            x2
        )

        y2 = min(
            image_height - 1,
            y2
        )

        class_name = (
            CLASS_NAMES[class_id]
        )


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # -------------------------------------------------
        # Label background
        # -------------------------------------------------

        text = (
            f"{class_id}: "
            f"{class_name}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        label_y = max(
            y1,
            text_height + 10
        )

        cv2.rectangle(
            output,
            (
                x1,
                label_y
                - text_height
                - 8
            ),
            (
                x1
                + text_width
                + 8,
                label_y
            ),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            text,
            (
                x1 + 4,
                label_y - 4
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return output


# =========================================================
# ADD PANEL TITLE
# =========================================================

def add_title(
    image,
    title
):

    title_height = 55

    canvas = np.full(
        (
            image.shape[0]
            + title_height,
            image.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )

    canvas[
        title_height:,
        :
    ] = image

    cv2.putText(
        canvas,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return canvas


# =========================================================
# RESIZE PANELS TO SAME SIZE
# =========================================================

def resize_panel(
    image,
    target_width=500
):

    height, width = image.shape[:2]

    ratio = (
        target_width /
        width
    )

    new_height = int(
        height * ratio
    )

    return cv2.resize(
        image,
        (
            target_width,
            new_height
        )
    )


# =========================================================
# CREATE ONE DEFECT COMPARISON
# =========================================================

def create_defect_preview(
    class_id
):

    class_name = (
        CLASS_NAMES[class_id]
    )

    print("\n" + "=" * 80)
    print(
        f"DEFECT: {class_name}"
    )
    print("=" * 80)

    image_path, labels = (
        find_sample_for_class(
            class_id
        )
    )

    if image_path is None:

        print(
            f"No image found for "
            f"{class_name}"
        )

        return

    print(
        "Selected image:",
        image_path.name
    )

    # -----------------------------------------------------
    # Load original image
    # -----------------------------------------------------

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        print(
            "Could not read:",
            image_path
        )

        return


    # =====================================================
    # ORIGINAL PANEL
    # =====================================================

    original_labelled = draw_labels(
        original,
        labels,
        target_class=class_id
    )

    original_panel = add_title(
        original_labelled,
        "Original Image"
    )

    original_panel = resize_panel(
        original_panel
    )


    # =====================================================
    # PROCESSED PANELS
    # =====================================================

    panels = [
        original_panel
    ]


    for canny_thresh in CANNY_THRESHOLDS:

        processed = preprocess_image(
            original,
            canny_thresh
        )

        processed_labelled = (
            draw_labels(
                processed,
                labels,
                target_class=class_id
            )
        )

        panel = add_title(
            processed_labelled,
            (
                f"Canny {canny_thresh} + "
                f"Hough"
            )
        )

        panel = resize_panel(
            panel
        )

        panels.append(
            panel
        )


    # =====================================================
    # MAKE ALL PANELS SAME HEIGHT
    # =====================================================

    minimum_height = min(
        panel.shape[0]
        for panel in panels
    )

    resized_panels = []

    for panel in panels:

        resized_panel = cv2.resize(
            panel,
            (
                panel.shape[1],
                minimum_height
            )
        )

        resized_panels.append(
            resized_panel
        )


    # =====================================================
    # COMBINE PANELS HORIZONTALLY
    # =====================================================

    comparison = cv2.hconcat(
        resized_panels
    )


    # =====================================================
    # SAVE RESULT
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        f"{class_id}_{class_name}_comparison.jpg"
    )

    cv2.imwrite(
        str(output_path),
        comparison
    )



    individual_dir = (
        OUTPUT_DIR /
        class_name
    )

    individual_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            individual_dir /
            "original_labelled.jpg"
        ),
        original_labelled
    )


    for canny_thresh in CANNY_THRESHOLDS:

        processed = preprocess_image(
            original,
            canny_thresh
        )

        processed_labelled = draw_labels(
            processed,
            labels,
            target_class=class_id
        )

        cv2.imwrite(
            str(
                individual_dir /
                f"canny_{canny_thresh[0]}_{canny_thresh[1]}.jpg"
            ),
            processed_labelled
        )


    print(
        "Saved comparison:"
    )

    print(
        output_path
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 2 IMAGE-PROCESSING PREVIEW (CANNY + HOUGH)")
    print("=" * 80)

    print(
        "Hough Threshold:",
        HOUGH_THRESHOLD
    )

    print(
        "Hough Min Line Length:",
        HOUGH_MIN_LINE_LENGTH
    )

    print(
        "Hough Max Line Gap:",
        HOUGH_MAX_LINE_GAP
    )

    print(
        "Canny Thresholds (Lower, Upper):",
        CANNY_THRESHOLDS
    )



    for class_id in CLASS_NAMES:

        create_defect_preview(
            class_id
        )


    print("\n" + "=" * 80)
    print("PREVIEW COMPLETED")
    print("=" * 80)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()