from pathlib import Path
import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

SOURCE_IMAGES = DATASET_ROOT / "train" / "images"
SOURCE_LABELS = DATASET_ROOT / "train" / "labels"

OUTPUT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_cliplimit_preview"
)


# =========================================================
# FIXED PARAMETERS
# =========================================================

CLAHE_TILE_GRID_SIZE = (8, 8)

# Best Adaptive Gaussian parameters
BLOCK_SIZE = 41
ADAPTIVE_C = 2

# Parameter being investigated
CLIP_LIMITS = [
    1.0,
    2.0,
    3.0
]


# =========================================================
# CLASSES
# =========================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# READ YOLO LABELS
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

            labels.append({
                "class_id": int(parts[0]),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4])
            })

    return labels


# =========================================================
# GET IMAGE FILES
# =========================================================

def get_image_files():

    if not SOURCE_IMAGES.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n"
            f"{SOURCE_IMAGES}"
        )

    return sorted([
        path
        for path in SOURCE_IMAGES.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])


# =========================================================
# FIND ONE IMAGE FOR EACH CLASS
# =========================================================

def find_sample_for_class(
    target_class,
    image_files
):

    # Prefer image containing only that class
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

        classes = {
            label["class_id"]
            for label in labels
        }

        if (
            target_class in classes
            and len(classes) == 1
        ):

            return image_path, labels


    # Otherwise any image containing class
    for image_path in image_files:

        label_path = (
            SOURCE_LABELS /
            f"{image_path.stem}.txt"
        )

        labels = read_yolo_labels(
            label_path
        )

        if any(
            label["class_id"] == target_class
            for label in labels
        ):

            return image_path, labels

    return None, None


# =========================================================
# PREPROCESS
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

    return cv2.cvtColor(
        thresholded,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# DRAW LABELS
# =========================================================

def draw_labels(
    image,
    labels,
    target_class
):

    output = image.copy()

    height, width = (
        output.shape[:2]
    )

    for label in labels:

        if label["class_id"] != target_class:
            continue

        x_center = (
            label["x_center"]
            * width
        )

        y_center = (
            label["y_center"]
            * height
        )

        box_width = (
            label["width"]
            * width
        )

        box_height = (
            label["height"]
            * height
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
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        class_name = (
            CLASS_NAMES[target_class]
        )

        text = (
            f"{target_class}: "
            f"{class_name}"
        )

        cv2.putText(
            output,
            text,
            (
                x1,
                max(
                    y1 - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return output


# =========================================================
# ADD TITLE
# =========================================================

def add_title(
    image,
    title
):

    title_height = 60

    canvas = np.full(
        (
            image.shape[0] + title_height,
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
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return canvas


# =========================================================
# RESIZE
# =========================================================

def resize_panel(
    image,
    width=500
):

    ratio = (
        width /
        image.shape[1]
    )

    height = int(
        image.shape[0]
        * ratio
    )

    return cv2.resize(
        image,
        (width, height)
    )


# =========================================================
# CREATE CLASS PREVIEW
# =========================================================

def create_preview(
    class_id,
    image_files
):

    class_name = (
        CLASS_NAMES[class_id]
    )

    image_path, labels = (
        find_sample_for_class(
            class_id,
            image_files
        )
    )

    if image_path is None:

        print(
            f"No image found for "
            f"{class_name}"
        )

        return

    print(
        f"\n{class_name}: "
        f"{image_path.name}"
    )

    original = cv2.imread(
        str(image_path)
    )

    if original is None:
        return


    # =====================================================
    # OUTPUT FOLDER
    # =====================================================

    individual_dir = (
        OUTPUT_DIR /
        class_name
    )

    individual_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # ORIGINAL
    # =====================================================

    original_labelled = (
        draw_labels(
            original,
            labels,
            class_id
        )
    )

    cv2.imwrite(
        str(
            individual_dir /
            "original.jpg"
        ),
        original_labelled
    )

    panels = [
        resize_panel(
            add_title(
                original_labelled,
                "Original Image"
            )
        )
    ]


    # =====================================================
    # CLIP LIMIT 1.0 / 2.0 / 3.0
    # =====================================================

    for clip_limit in CLIP_LIMITS:

        processed = preprocess_image(
            original,
            clip_limit
        )

        processed_labelled = (
            draw_labels(
                processed,
                labels,
                class_id
            )
        )

        filename = (
            f"clip_{clip_limit:.1f}.jpg"
        )

        cv2.imwrite(
            str(
                individual_dir /
                filename
            ),
            processed_labelled
        )

        title = (
            f"clipLimit = "
            f"{clip_limit:.1f}"
        )

        panels.append(
            resize_panel(
                add_title(
                    processed_labelled,
                    title
                )
            )
        )


    # =====================================================
    # NORMALISE HEIGHT
    # =====================================================

    min_height = min(
        panel.shape[0]
        for panel in panels
    )

    panels = [
        cv2.resize(
            panel,
            (
                panel.shape[1],
                min_height
            )
        )
        for panel in panels
    ]


    comparison = cv2.hconcat(
        panels
    )

    comparison_path = (
        OUTPUT_DIR /
        (
            f"{class_id}_"
            f"{class_name}_"
            f"cliplimit_comparison.jpg"
        )
    )

    cv2.imwrite(
        str(comparison_path),
        comparison
    )

    print(
        f"Saved: "
        f"{comparison_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("STUDENT 1 - CLAHE CLIPLIMIT PREVIEW")
    print("=" * 80)

    print(
        f"Fixed blockSize : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Fixed C         : "
        f"{ADAPTIVE_C}"
    )

    print(
        f"Tile Grid       : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Clip Limits     : "
        f"{CLIP_LIMITS}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = (
        get_image_files()
    )

    for class_id in CLASS_NAMES:

        create_preview(
            class_id,
            image_files
        )

    print("\n" + "=" * 80)
    print("PREVIEW COMPLETED")
    print("=" * 80)

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()