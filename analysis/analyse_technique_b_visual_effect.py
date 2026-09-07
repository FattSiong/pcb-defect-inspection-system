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
    r"D:\pcb-defect-inspection-system"
)

DATASET_ROOT = Path(
    r"C:\Users\user\Downloads\pcb_dataset_reduced"
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
    / "technique_b_selection_analysis"
)

CANNY_OUTPUT = (
    OUTPUT_ROOT
    / "01_canny_analysis"
)

HOUGH_OUTPUT = (
    OUTPUT_ROOT
    / "02_hough_analysis"
)

CANNY_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

HOUGH_OUTPUT.mkdir(
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
# PROCESSING PARAMETERS (FINAL TECHNIQUE B WINNERS)
# ============================================================

CANNY_THRESH = (50, 150)
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 5

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180


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


        if class_id == target_class_id:

            colour = (
                0,
                0,
                255
            )

            thickness = 4


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
# CANNY EDGE DETECTION
# ============================================================

def apply_canny(
    original_image
):

    if len(original_image.shape) == 3:
        grayscale = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        grayscale = original_image

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    canny_edges = cv2.Canny(
        blurred,
        CANNY_THRESH[0],
        CANNY_THRESH[1]
    )

    return (
        grayscale,
        canny_edges
    )


# ============================================================
# HOUGH TRANSFORM
# ============================================================

def apply_hough(
    canny_edges
):

    lines = cv2.HoughLinesP(
        canny_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    hough_image = cv2.cvtColor(
        canny_edges,
        cv2.COLOR_GRAY2BGR
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                hough_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return hough_image


# ============================================================
# BGR -> RGB
# ============================================================

def bgr_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# INDIVIDUAL CANNY COMPARISON
# ============================================================

def save_individual_canny_comparison(
    class_name,
    original_labelled,
    canny_labelled
):

    output_dir = (
        CANNY_OUTPUT
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
            / "02_canny_labelled.png"
        ),
        canny_labelled
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

    figure.suptitle(
        f"Canny Edge Detection Analysis - {display_name}",
        fontsize=20,
        fontweight="bold",
        y=0.97
    )



    axes[0].set_title(
        "Original PCB Image",
        fontsize=16,
        pad=12
    )



    axes[1].set_title(
        "After Canny Edge Detection",
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
            canny_labelled
        )
    )


    for axis in axes:

        axis.axis(
            "off"
        )



    figure.subplots_adjust(
        top=0.82,
        bottom=0.04,
        left=0.03,
        right=0.97,
        wspace=0.08
    )


    comparison_path = (
        output_dir
        / "canny_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return comparison_path


# ============================================================
# INDIVIDUAL HOUGH COMPARISON
# ============================================================

def save_individual_hough_comparison(
    class_name,
    canny_labelled,
    hough_labelled
):

    output_dir = (
        HOUGH_OUTPUT
        / class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_canny_labelled.png"
        ),
        canny_labelled
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_canny_hough_labelled.png"
        ),
        hough_labelled
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



    figure.suptitle(
        (
            "Hough Transform Analysis "
            f"- {display_name}"
        ),
        fontsize=20,
        fontweight="bold",
        y=0.97
    )



    axes[0].set_title(
        "After Canny Edge Detection",
        fontsize=16,
        pad=12
    )



    axes[1].set_title(
        (
            "After Canny +\n"
            "Hough Transform"
        ),
        fontsize=16,
        pad=12
    )


    axes[0].imshow(
        bgr_to_rgb(
            canny_labelled
        )
    )

    axes[1].imshow(
        bgr_to_rgb(
            hough_labelled
        )
    )


    for axis in axes:

        axis.axis(
            "off"
        )



    figure.subplots_adjust(
        top=0.78,
        bottom=0.04,
        left=0.03,
        right=0.97,
        wspace=0.08
    )


    comparison_path = (
        output_dir
        / "hough_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return comparison_path


# ============================================================
# COMBINED CANNY FIGURE
# ============================================================

def create_combined_canny_figure(
    results
):

    figure, axes = plt.subplots(
        6,
        2,
        figsize=(14, 26)
    )


    figure.suptitle(
        (
            "Visual Effect of Canny Edge Detection Across "
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
        "After Canny Edge Detection",
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
                    "canny_labelled"
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
        CANNY_OUTPUT
        / "canny_all_defects_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# COMBINED HOUGH FIGURE
# ============================================================

def create_combined_hough_figure(
    results
):

    figure, axes = plt.subplots(
        6,
        2,
        figsize=(14, 26)
    )


    figure.suptitle(
        (
            "Visual Effect of Hough Transform "
            "After Canny Edge Detection Across Six PCB Defect Classes"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.99
    )


    axes[0, 0].set_title(
        "After Canny Edge Detection",
        fontsize=15,
        pad=12
    )


    axes[0, 1].set_title(
        (
            "After Canny +\n"
            "Hough Transform"
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
                    "canny_labelled"
                ]
            )
        )

        axes[
            row_index,
            1
        ].imshow(
            bgr_to_rgb(
                result[
                    "hough_labelled"
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
        HOUGH_OUTPUT
        / "canny_hough_all_defects_comparison.png"
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
        "TECHNIQUE B PRELIMINARY "
        "TECHNIQUE SELECTION ANALYSIS"
    )
    print("=" * 80)

    print()
    print(
        "Processing pipeline:"
    )

    print(
        "Original -> Canny Edge Detection -> "
        "Hough Transform"
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
        # ORIGINAL -> CANNY
        # ----------------------------------------------------

        (
            grayscale,
            canny_image
        ) = apply_canny(
            original
        )


        # ----------------------------------------------------
        # CANNY -> HOUGH
        # ----------------------------------------------------

        hough_image = (
            apply_hough(
                canny_image
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

        canny_labelled = (
            draw_labels(
                canny_image,
                labels,
                class_id
            )
        )

        hough_labelled = (
            draw_labels(
                hough_image,
                labels,
                class_id
            )
        )


        # ----------------------------------------------------
        # SAVE INDIVIDUAL FIGURES
        # ----------------------------------------------------

        canny_path = (
            save_individual_canny_comparison(
                class_name,
                original_labelled,
                canny_labelled
            )
        )


        hough_path = (
            save_individual_hough_comparison(
                class_name,
                canny_labelled,
                hough_labelled
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

                "canny_labelled":
                    canny_labelled,

                "hough_labelled":
                    hough_labelled,
            }
        )


        print(
            f"Canny Edge figure saved:\n"
            f"{canny_path}"
        )

        print()

        print(
            f"Hough Transform figure saved:\n"
            f"{hough_path}"
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


    canny_combined_path = (
        create_combined_canny_figure(
            results
        )
    )


    hough_combined_path = (
        create_combined_hough_figure(
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
        "Combined Canny Edge figure:"
    )

    print(
        canny_combined_path
    )

    print()

    print(
        "Combined Hough Transform figure:"
    )

    print(
        hough_combined_path
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