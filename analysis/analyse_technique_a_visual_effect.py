from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

DATA_SPLIT = "train"

IMAGE_DIR = (
    DATASET_ROOT
    / DATA_SPLIT
    / "images"
)

LABEL_DIR = (
    DATASET_ROOT
    / DATA_SPLIT
    / "labels"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "technique_a_selection_analysis"
)

CLAHE_OUTPUT = (
    OUTPUT_ROOT
    / "01_clahe_analysis"
)

ADAPTIVE_OUTPUT = (
    OUTPUT_ROOT
    / "02_adaptive_gaussian_analysis"
)

CLAHE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

ADAPTIVE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEFECT CLASSES
# ============================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# ============================================================
# PROCESSING PARAMETERS
# ============================================================

CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ADAPTIVE_BLOCK_SIZE = 41
ADAPTIVE_C = 2


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
]


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(stem):

    for extension in IMAGE_EXTENSIONS:

        image_path = (
            IMAGE_DIR
            / f"{stem}{extension}"
        )

        if image_path.exists():
            return image_path

    return None


# ============================================================
# READ YOLO LABELS
# ============================================================

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

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            labels.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )

    return labels


# ============================================================
# SELECT REPRESENTATIVE IMAGE
# ============================================================

def select_representative_image(
    target_class_id
):

    candidates = []

    for label_path in LABEL_DIR.glob("*.txt"):

        labels = read_yolo_labels(
            label_path
        )

        target_labels = [
            label
            for label in labels
            if label["class_id"]
            == target_class_id
        ]

        if not target_labels:
            continue

        image_path = find_image(
            label_path.stem
        )

        if image_path is None:
            continue

        target_label = max(
            target_labels,
            key=lambda item:
                item["area"]
        )

        candidates.append(
            {
                "image_path":
                    image_path,

                "label_path":
                    label_path,

                "labels":
                    labels,

                "target_label":
                    target_label,

                "target_area":
                    target_label["area"],
            }
        )

    if not candidates:

        raise RuntimeError(
            f"No image found for class "
            f"{target_class_id}"
        )

    areas = [
        candidate["target_area"]
        for candidate in candidates
    ]

    median_area = float(
        np.median(areas)
    )

    selected = min(
        candidates,
        key=lambda candidate:
            abs(
                candidate["target_area"]
                - median_area
            )
    )

    return selected


# ============================================================
# YOLO BOX TO PIXEL COORDINATES
# ============================================================

def yolo_to_pixel_box(
    label,
    image_width,
    image_height
):

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

    x1 = max(
        0,
        min(
            image_width - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            y1
        )
    )

    x2 = max(
        0,
        min(
            image_width - 1,
            x2
        )
    )

    y2 = max(
        0,
        min(
            image_height - 1,
            y2
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# DRAW DEFECT LABELS
# ============================================================

def draw_labels(
    image,
    labels,
    target_class_id
):

    if len(image.shape) == 2:

        labelled_image = (
            cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )
        )

    else:

        labelled_image = (
            image.copy()
        )

    image_height, image_width = (
        labelled_image.shape[:2]
    )

    for label in labels:

        class_id = (
            label["class_id"]
        )

        (
            x1,
            y1,
            x2,
            y2
        ) = yolo_to_pixel_box(
            label,
            image_width,
            image_height
        )

        class_name = (
            CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            )
        )

        display_name = (
            class_name
            .replace(
                "_",
                " "
            )
            .title()
        )

        # Target class = red
        if class_id == target_class_id:

            colour = (
                0,
                0,
                255
            )

            thickness = 4

        # Other classes = green
        else:

            colour = (
                0,
                255,
                0
            )

            thickness = 2

        cv2.rectangle(
            labelled_image,
            (x1, y1),
            (x2, y2),
            colour,
            thickness
        )

        font_scale = max(
            0.45,
            min(
                image_width,
                image_height
            ) / 1200
        )

        text_thickness = max(
            1,
            int(
                font_scale * 2
            )
        )

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            display_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness
        )

        text_x = x1

        text_y = (
            y1 - 8
        )

        if (
            text_y
            - text_height
            < 0
        ):

            text_y = (
                y1
                + text_height
                + 8
            )

        background_y1 = max(
            0,
            text_y
            - text_height
            - 5
        )

        background_y2 = min(
            image_height - 1,
            text_y + 5
        )

        background_x2 = min(
            image_width - 1,
            text_x
            + text_width
            + 8
        )

        cv2.rectangle(
            labelled_image,
            (
                text_x,
                background_y1
            ),
            (
                background_x2,
                background_y2
            ),
            colour,
            -1
        )

        cv2.putText(
            labelled_image,
            display_name,
            (
                text_x + 4,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (
                255,
                255,
                255
            ),
            text_thickness,
            cv2.LINE_AA
        )

    return labelled_image


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    original_image
):

    grayscale = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    clahe_image = clahe.apply(
        grayscale
    )

    return (
        grayscale,
        clahe_image
    )


# ============================================================
# ADAPTIVE GAUSSIAN THRESHOLDING
# ============================================================

def apply_adaptive_gaussian(
    clahe_image
):

    adaptive_image = (
        cv2.adaptiveThreshold(
            clahe_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C
        )
    )

    return adaptive_image


# ============================================================
# BGR -> RGB
# ============================================================

def bgr_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# INDIVIDUAL CLAHE COMPARISON
# ============================================================

def save_individual_clahe_comparison(
    class_name,
    original_labelled,
    clahe_labelled
):

    output_dir = (
        CLAHE_OUTPUT
        / class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_original_labelled.png"
        ),
        original_labelled
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_clahe_labelled.png"
        ),
        clahe_labelled
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8)
    )


    # Main figure title
    display_name = (
        class_name
        .replace(
            "_",
            " "
        )
        .title()
    )

    figure.suptitle(
        f"CLAHE Analysis - {display_name}",
        fontsize=20,
        fontweight="bold",
        y=0.97
    )


    # Left title
    axes[0].set_title(
        "Original PCB Image",
        fontsize=16,
        pad=12
    )


    # Right title
    axes[1].set_title(
        "After CLAHE",
        fontsize=16,
        pad=12
    )


    axes[0].imshow(
        bgr_to_rgb(
            original_labelled
        )
    )

    axes[1].imshow(
        bgr_to_rgb(
            clahe_labelled
        )
    )


    for axis in axes:

        axis.axis(
            "off"
        )


    # IMPORTANT:
    # Reserve space at the top for the main title
    figure.subplots_adjust(
        top=0.82,
        bottom=0.04,
        left=0.03,
        right=0.97,
        wspace=0.08
    )


    comparison_path = (
        output_dir
        / "clahe_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return comparison_path


# ============================================================
# INDIVIDUAL ADAPTIVE GAUSSIAN COMPARISON
# ============================================================

def save_individual_adaptive_comparison(
    class_name,
    clahe_labelled,
    adaptive_labelled
):

    output_dir = (
        ADAPTIVE_OUTPUT
        / class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_clahe_labelled.png"
        ),
        clahe_labelled
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_clahe_adaptive_gaussian_labelled.png"
        ),
        adaptive_labelled
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8)
    )


    display_name = (
        class_name
        .replace(
            "_",
            " "
        )
        .title()
    )


    # Main figure title
    figure.suptitle(
        (
            "Adaptive Gaussian Thresholding Analysis "
            f"- {display_name}"
        ),
        fontsize=20,
        fontweight="bold",
        y=0.97
    )


    # Left title
    axes[0].set_title(
        "After CLAHE",
        fontsize=16,
        pad=12
    )


    # Right title - split into two lines
    axes[1].set_title(
        (
            "After CLAHE +\n"
            "Adaptive Gaussian Thresholding"
        ),
        fontsize=16,
        pad=12
    )


    axes[0].imshow(
        bgr_to_rgb(
            clahe_labelled
        )
    )

    axes[1].imshow(
        bgr_to_rgb(
            adaptive_labelled
        )
    )


    for axis in axes:

        axis.axis(
            "off"
        )


    # Reserve enough space between:
    # main title
    # subplot titles
    # images
    figure.subplots_adjust(
        top=0.78,
        bottom=0.04,
        left=0.03,
        right=0.97,
        wspace=0.08
    )


    comparison_path = (
        output_dir
        / "adaptive_gaussian_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return comparison_path


# ============================================================
# COMBINED CLAHE FIGURE
# ============================================================

def create_combined_clahe_figure(
    results
):

    figure, axes = plt.subplots(
        6,
        2,
        figsize=(14, 26)
    )


    figure.suptitle(
        (
            "Visual Effect of CLAHE Across "
            "Six PCB Defect Classes"
        ),
        fontsize=20,
        fontweight="bold",
        y=0.985
    )


    axes[0, 0].set_title(
        "Original PCB Image",
        fontsize=15,
        pad=12
    )

    axes[0, 1].set_title(
        "After CLAHE",
        fontsize=15,
        pad=12
    )


    for row_index, result in enumerate(
        results
    ):

        axes[
            row_index,
            0
        ].imshow(
            bgr_to_rgb(
                result[
                    "original_labelled"
                ]
            )
        )

        axes[
            row_index,
            1
        ].imshow(
            bgr_to_rgb(
                result[
                    "clahe_labelled"
                ]
            )
        )

        defect_name = (
            result[
                "class_name"
            ]
            .replace(
                "_",
                " "
            )
            .title()
        )

        axes[
            row_index,
            0
        ].set_ylabel(
            defect_name,
            fontsize=12,
            labelpad=15
        )

        axes[
            row_index,
            0
        ].axis(
            "off"
        )

        axes[
            row_index,
            1
        ].axis(
            "off"
        )


    figure.subplots_adjust(
        top=0.94,
        bottom=0.02,
        left=0.05,
        right=0.98,
        hspace=0.08,
        wspace=0.08
    )


    output_path = (
        CLAHE_OUTPUT
        / "clahe_all_defects_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# COMBINED ADAPTIVE GAUSSIAN FIGURE
# ============================================================

def create_combined_adaptive_figure(
    results
):

    figure, axes = plt.subplots(
        6,
        2,
        figsize=(14, 26)
    )


    figure.suptitle(
        (
            "Visual Effect of Adaptive Gaussian Thresholding "
            "After CLAHE Across Six PCB Defect Classes"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.99
    )


    axes[0, 0].set_title(
        "After CLAHE",
        fontsize=15,
        pad=12
    )


    axes[0, 1].set_title(
        (
            "After CLAHE +\n"
            "Adaptive Gaussian Thresholding"
        ),
        fontsize=15,
        pad=12
    )


    for row_index, result in enumerate(
        results
    ):

        axes[
            row_index,
            0
        ].imshow(
            bgr_to_rgb(
                result[
                    "clahe_labelled"
                ]
            )
        )

        axes[
            row_index,
            1
        ].imshow(
            bgr_to_rgb(
                result[
                    "adaptive_labelled"
                ]
            )
        )

        defect_name = (
            result[
                "class_name"
            ]
            .replace(
                "_",
                " "
            )
            .title()
        )

        axes[
            row_index,
            0
        ].set_ylabel(
            defect_name,
            fontsize=12,
            labelpad=15
        )

        axes[
            row_index,
            0
        ].axis(
            "off"
        )

        axes[
            row_index,
            1
        ].axis(
            "off"
        )


    figure.subplots_adjust(
        top=0.93,
        bottom=0.02,
        left=0.05,
        right=0.98,
        hspace=0.08,
        wspace=0.08
    )


    output_path = (
        ADAPTIVE_OUTPUT
        / "clahe_adaptive_gaussian_all_defects_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "TECHNIQUE A PRELIMINARY "
        "TECHNIQUE SELECTION ANALYSIS"
    )
    print("=" * 80)

    print()
    print(
        "Processing pipeline:"
    )

    print(
        "Original -> CLAHE -> "
        "Adaptive Gaussian Thresholding"
    )

    results = []


    # ========================================================
    # PROCESS SIX DEFECT CLASSES
    # ========================================================

    for (
        class_id,
        class_name
    ) in CLASS_NAMES.items():

        print()
        print("-" * 80)

        print(
            f"Processing: "
            f"{class_name}"
        )


        selected = (
            select_representative_image(
                class_id
            )
        )


        image_path = (
            selected[
                "image_path"
            ]
        )

        labels = (
            selected[
                "labels"
            ]
        )


        print(
            f"Selected image: "
            f"{image_path.name}"
        )


        # ----------------------------------------------------
        # LOAD ORIGINAL
        # ----------------------------------------------------

        original = cv2.imread(
            str(
                image_path
            )
        )

        if original is None:

            raise RuntimeError(
                f"Unable to read:\n"
                f"{image_path}"
            )


        # ----------------------------------------------------
        # ORIGINAL -> CLAHE
        # ----------------------------------------------------

        (
            grayscale,
            clahe_image
        ) = apply_clahe(
            original
        )


        # ----------------------------------------------------
        # CLAHE -> ADAPTIVE GAUSSIAN
        # ----------------------------------------------------

        adaptive_image = (
            apply_adaptive_gaussian(
                clahe_image
            )
        )


        # ----------------------------------------------------
        # DRAW LABELS
        # ----------------------------------------------------

        original_labelled = (
            draw_labels(
                original,
                labels,
                class_id
            )
        )

        clahe_labelled = (
            draw_labels(
                clahe_image,
                labels,
                class_id
            )
        )

        adaptive_labelled = (
            draw_labels(
                adaptive_image,
                labels,
                class_id
            )
        )


        # ----------------------------------------------------
        # SAVE INDIVIDUAL FIGURES
        # ----------------------------------------------------

        clahe_path = (
            save_individual_clahe_comparison(
                class_name,
                original_labelled,
                clahe_labelled
            )
        )


        adaptive_path = (
            save_individual_adaptive_comparison(
                class_name,
                clahe_labelled,
                adaptive_labelled
            )
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        results.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "image_name":
                    image_path.name,

                "original_labelled":
                    original_labelled,

                "clahe_labelled":
                    clahe_labelled,

                "adaptive_labelled":
                    adaptive_labelled,
            }
        )


        print(
            f"CLAHE figure saved:\n"
            f"{clahe_path}"
        )

        print()

        print(
            f"Adaptive figure saved:\n"
            f"{adaptive_path}"
        )


    # ========================================================
    # COMBINED FIGURES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GENERATING COMBINED FIGURES"
    )
    print("=" * 80)


    clahe_combined_path = (
        create_combined_clahe_figure(
            results
        )
    )


    adaptive_combined_path = (
        create_combined_adaptive_figure(
            results
        )
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "ANALYSIS COMPLETED"
    )
    print("=" * 80)

    print()

    print(
        "Combined CLAHE figure:"
    )

    print(
        clahe_combined_path
    )

    print()

    print(
        "Combined Adaptive Gaussian figure:"
    )

    print(
        adaptive_combined_path
    )

    print()

    print(
        f"All outputs saved to:\n"
        f"{OUTPUT_ROOT}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()