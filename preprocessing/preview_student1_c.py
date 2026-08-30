from pathlib import Path
import cv2
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

SOURCE_IMAGES = (
    DATASET_ROOT /
    "train" /
    "images"
)

SOURCE_LABELS = (
    DATASET_ROOT /
    "train" /
    "labels"
)

OUTPUT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_c_preview"
)


# =========================================================
# FIXED STUDENT 1 PARAMETERS
# =========================================================

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (
    8,
    8
)

# Best block size from previous experiment
BLOCK_SIZE = 41

# Parameter currently being investigated
C_VALUES = [
    2,
    5,
    8
]


# =========================================================
# DEFECT CLASSES
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
# SUPPORTED IMAGE TYPES
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# READ YOLO LABEL FILE
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

            parts = (
                line
                .strip()
                .split()
            )

            if len(parts) != 5:
                continue

            labels.append({
                "class_id":
                    int(parts[0]),

                "x_center":
                    float(parts[1]),

                "y_center":
                    float(parts[2]),

                "width":
                    float(parts[3]),

                "height":
                    float(parts[4])
            })

    return labels


# =========================================================
# FIND MATCHING LABEL FILE
# =========================================================

def find_label_file(image_path):

    # Normal YOLO naming:
    # image.jpg -> image.txt

    normal_label = (
        SOURCE_LABELS /
        f"{image_path.stem}.txt"
    )

    if normal_label.exists():
        return normal_label

    return None


# =========================================================
# GET ALL IMAGE FILES
# =========================================================

def get_image_files():

    if not SOURCE_IMAGES.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n"
            f"{SOURCE_IMAGES}"
        )

    return sorted(
        [
            path
            for path in SOURCE_IMAGES.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


# =========================================================
# FIND ONE SAMPLE FOR EACH DEFECT CLASS
# =========================================================

def find_sample_for_class(
    target_class,
    image_files
):

    # -----------------------------------------------------
    # First preference:
    # image contains only the target defect class
    # -----------------------------------------------------

    for image_path in image_files:

        label_path = find_label_file(
            image_path
        )

        if label_path is None:
            continue

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
            target_class
            in unique_classes
            and len(unique_classes) == 1
        ):

            return (
                image_path,
                labels
            )


    # -----------------------------------------------------
    # Fallback:
    # use any image containing target class
    # -----------------------------------------------------

    for image_path in image_files:

        label_path = find_label_file(
            image_path
        )

        if label_path is None:
            continue

        labels = read_yolo_labels(
            label_path
        )

        if not labels:
            continue

        unique_classes = {
            label["class_id"]
            for label in labels
        }

        if target_class in unique_classes:

            return (
                image_path,
                labels
            )


    return (
        None,
        None
    )


# =========================================================
# APPLY CLAHE + ADAPTIVE GAUSSIAN THRESHOLDING
# =========================================================

def preprocess_image(
    image,
    c_value
):

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
    # -----------------------------------------------------

    thresholded = cv2.adaptiveThreshold(

        enhanced,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        BLOCK_SIZE,

        c_value
    )


    # Convert back to BGR so coloured labels
    # can be drawn
    thresholded_bgr = cv2.cvtColor(
        thresholded,
        cv2.COLOR_GRAY2BGR
    )

    return thresholded_bgr


# =========================================================
# DRAW DEFECT BOUNDING BOX
# =========================================================

def draw_labels(
    image,
    labels,
    target_class
):

    output = image.copy()

    image_height, image_width = (
        output.shape[:2]
    )


    for label in labels:

        class_id = label[
            "class_id"
        ]

        # Only display the defect class
        # currently being previewed
        if class_id != target_class:
            continue


        # -------------------------------------------------
        # Convert normalised YOLO coordinates
        # into pixel coordinates
        # -------------------------------------------------

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


        # Keep box inside image
        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image_width - 1,
            x2
        )

        y2 = min(
            image_height - 1,
            y2
        )


        # -------------------------------------------------
        # Draw bounding box
        # -------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )


        # -------------------------------------------------
        # Defect label
        # -------------------------------------------------

        class_name = (
            CLASS_NAMES[class_id]
        )

        text = (
            f"{class_id}: "
            f"{class_name}"
        )


        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        font_scale = 0.55

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


        text_y = max(
            y1,
            text_height + 10
        )


        # Label background
        cv2.rectangle(
            output,

            (
                x1,
                text_y
                - text_height
                - 8
            ),

            (
                x1
                + text_width
                + 8,
                text_y
            ),

            (0, 0, 255),

            -1
        )


        # White text
        cv2.putText(
            output,

            text,

            (
                x1 + 4,
                text_y - 4
            ),

            font,

            font_scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


    return output


# =========================================================
# ADD TITLE ABOVE PANEL
# =========================================================

def add_title(
    image,
    title
):

    title_height = 60


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

        (
            15,
            38
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 0, 0),

        2,

        cv2.LINE_AA
    )


    return canvas


# =========================================================
# RESIZE PANEL
# =========================================================

def resize_panel(
    image,
    target_width=500
):

    height, width = (
        image.shape[:2]
    )


    ratio = (
        target_width /
        width
    )


    target_height = int(
        height * ratio
    )


    return cv2.resize(
        image,
        (
            target_width,
            target_height
        )
    )


# =========================================================
# NORMALISE PANEL HEIGHTS
# =========================================================

def normalise_panel_heights(
    panels
):

    minimum_height = min(
        panel.shape[0]
        for panel in panels
    )


    output = []


    for panel in panels:

        resized = cv2.resize(
            panel,

            (
                panel.shape[1],
                minimum_height
            )
        )

        output.append(
            resized
        )


    return output


# =========================================================
# CREATE PREVIEW FOR ONE DEFECT
# =========================================================

def create_defect_preview(
    class_id,
    image_files
):

    class_name = (
        CLASS_NAMES[class_id]
    )


    print("\n" + "=" * 80)

    print(
        f"DEFECT: {class_name}"
    )

    print("=" * 80)


    (
        image_path,
        labels
    ) = find_sample_for_class(
        class_id,
        image_files
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
    # Load source image
    # -----------------------------------------------------

    original = cv2.imread(
        str(image_path)
    )


    if original is None:

        print(
            f"Could not read image:\n"
            f"{image_path}"
        )

        return


    # =====================================================
    # ORIGINAL PANEL
    # =====================================================

    original_labelled = draw_labels(
        original,
        labels,
        class_id
    )


    original_panel = add_title(
        original_labelled,
        "Original Image"
    )


    original_panel = resize_panel(
        original_panel
    )


    panels = [
        original_panel
    ]


    # =====================================================
    # OUTPUT DIRECTORY FOR THIS DEFECT
    # =====================================================

    individual_dir = (
        OUTPUT_DIR /
        class_name
    )


    individual_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save labelled original
    cv2.imwrite(
        str(
            individual_dir /
            "original_labelled.jpg"
        ),
        original_labelled
    )


    # =====================================================
    # C = 2, 5, 8
    # =====================================================

    for c_value in C_VALUES:

        processed = preprocess_image(
            original,
            c_value
        )


        processed_labelled = draw_labels(
            processed,
            labels,
            class_id
        )


        panel = add_title(
            processed_labelled,
            (
                f"C = {c_value} "
                f"(blockSize = {BLOCK_SIZE})"
            )
        )


        panel = resize_panel(
            panel
        )


        panels.append(
            panel
        )


        # Save individual processed image
        cv2.imwrite(
            str(
                individual_dir /
                f"c_{c_value}.jpg"
            ),
            processed_labelled
        )


    # =====================================================
    # COMBINE ALL PANELS
    # =====================================================

    panels = normalise_panel_heights(
        panels
    )


    comparison = cv2.hconcat(
        panels
    )


    # =====================================================
    # SAVE COMPARISON
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    comparison_path = (
        OUTPUT_DIR /
        (
            f"{class_id}_"
            f"{class_name}_"
            f"c_comparison.jpg"
        )
    )


    cv2.imwrite(
        str(comparison_path),
        comparison
    )


    print(
        "Saved comparison:"
    )

    print(
        comparison_path
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 - C PARAMETER IMAGE PREVIEW")
    print("=" * 80)


    print(
        f"\nDataset:\n"
        f"{DATASET_ROOT}"
    )


    print(
        f"\nCLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tileGrid  : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Block Size      : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C values        : "
        f"{C_VALUES}"
    )


    image_files = get_image_files()


    print(
        f"\nTraining images found: "
        f"{len(image_files)}"
    )


    # -----------------------------------------------------
    # Create one preview for each defect class
    # -----------------------------------------------------

    for class_id in CLASS_NAMES:

        create_defect_preview(
            class_id,
            image_files
        )


    print("\n" + "=" * 80)
    print("C PARAMETER PREVIEW COMPLETED")
    print("=" * 80)


    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()