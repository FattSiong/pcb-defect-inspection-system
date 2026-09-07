"""Create report-ready visual comparisons for the B1-M3-W2 pipeline."""

from pathlib import Path
import sys

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow this file to be run directly from the analysis directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.bilateral_morphology import enhance_pcb


DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "pcb_dataset_reduced"
)
DATA_SPLIT = "train"
IMAGE_DIR = DATASET_ROOT / DATA_SPLIT / "images"
LABEL_DIR = DATASET_ROOT / DATA_SPLIT / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_morphology_visual_analysis"
)
BILATERAL_OUTPUT = OUTPUT_ROOT / "01_bilateral_analysis"
MORPHOLOGY_OUTPUT = OUTPUT_ROOT / "02_morphology_analysis"


# ============================================================
# LOCKED B1-M3-W2 PARAMETERS
# ============================================================

BILATERAL_D = 3
SIGMA_COLOR = 15
SIGMA_SPACE = 3
KERNEL_SIZES = (7, 15, 31)
ALPHA = 0.7
BETA = 0.7

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


# ============================================================
# DATASET AND LABEL HELPERS
# ============================================================

def check_inputs():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n{LABEL_DIR}"
        )

    image_count = sum(
        1
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_count = sum(1 for _ in LABEL_DIR.glob("*.txt"))

    if image_count == 0 or label_count == 0:
        raise RuntimeError("The selected dataset split is empty.")

    BILATERAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    MORPHOLOGY_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Dataset        : {DATASET_ROOT}")
    print(f"Split          : {DATA_SPLIT}")
    print(f"Images         : {image_count}")
    print(f"Labels         : {label_count}")
    print(f"Bilateral d    : {BILATERAL_D}")
    print(f"Sigma colour   : {SIGMA_COLOR}")
    print(f"Sigma space    : {SIGMA_SPACE}")
    print(f"Kernels        : {KERNEL_SIZES}")
    print(f"Alpha          : {ALPHA}")
    print(f"Beta           : {BETA}")
    print(f"Output         : {OUTPUT_ROOT}")


def find_image(stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGE_DIR / f"{stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def read_yolo_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if len(values) < 5:
                continue

            class_id = int(values[0])
            width = float(values[3])
            height = float(values[4])

            labels.append(
                {
                    "class_id": class_id,
                    "x_center": float(values[1]),
                    "y_center": float(values[2]),
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )

    return labels


def select_representative_image(target_class_id):
    candidates = []

    for label_path in LABEL_DIR.glob("*.txt"):
        labels = read_yolo_labels(label_path)
        target_labels = [
            label
            for label in labels
            if label["class_id"] == target_class_id
        ]

        if not target_labels:
            continue

        image_path = find_image(label_path.stem)

        if image_path is None:
            continue

        target_label = max(
            target_labels,
            key=lambda item: item["area"],
        )
        candidates.append(
            {
                "image_path": image_path,
                "labels": labels,
                "target_area": target_label["area"],
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No image found for class {target_class_id}."
        )

    median_area = float(
        np.median(
            [candidate["target_area"] for candidate in candidates]
        )
    )

    return min(
        candidates,
        key=lambda candidate: abs(
            candidate["target_area"] - median_area
        ),
    )


# ============================================================
# DRAW GROUND-TRUTH BOXES
# ============================================================

def yolo_to_pixel_box(label, image_width, image_height):
    x_center = label["x_center"] * image_width
    y_center = label["y_center"] * image_height
    box_width = label["width"] * image_width
    box_height = label["height"] * image_height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    return (
        max(0, min(image_width - 1, x1)),
        max(0, min(image_height - 1, y1)),
        max(0, min(image_width - 1, x2)),
        max(0, min(image_height - 1, y2)),
    )


def draw_labels(image, labels, target_class_id):
    if image.ndim == 2:
        labelled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        labelled = image.copy()

    image_height, image_width = labelled.shape[:2]

    for label in labels:
        class_id = label["class_id"]
        x1, y1, x2, y2 = yolo_to_pixel_box(
            label,
            image_width,
            image_height,
        )
        display_name = (
            CLASS_NAMES.get(class_id, f"class_{class_id}")
            .replace("_", " ")
            .title()
        )

        if class_id == target_class_id:
            colour = (0, 0, 255)
            thickness = 4
        else:
            colour = (0, 255, 0)
            thickness = 2

        cv2.rectangle(
            labelled,
            (x1, y1),
            (x2, y2),
            colour,
            thickness,
        )

        font_scale = max(
            0.45,
            min(image_width, image_height) / 1200,
        )
        text_thickness = max(1, int(font_scale * 2))
        (text_width, text_height), _ = cv2.getTextSize(
            display_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_y = y1 - 8

        if text_y - text_height < 0:
            text_y = y1 + text_height + 8

        background_y1 = max(0, text_y - text_height - 5)
        background_y2 = min(image_height - 1, text_y + 5)
        background_x2 = min(
            image_width - 1,
            x1 + text_width + 8,
        )

        cv2.rectangle(
            labelled,
            (x1, background_y1),
            (background_x2, background_y2),
            colour,
            -1,
        )
        cv2.putText(
            labelled,
            display_name,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return labelled


# ============================================================
# B1-M3-W2 PIPELINE
# ============================================================

def process_image(original):
    maps = enhance_pcb(
        original,
        bilateral_d=BILATERAL_D,
        sigma_color=SIGMA_COLOR,
        sigma_space=SIGMA_SPACE,
        kernel_sizes=KERNEL_SIZES,
        alpha=ALPHA,
        beta=BETA,
        return_maps=True,
    )

    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    _, channel_a, channel_b = cv2.split(original_lab)
    bilateral_lab = cv2.merge(
        (maps["filtered_lightness"], channel_a, channel_b)
    )
    bilateral_image = cv2.cvtColor(
        bilateral_lab,
        cv2.COLOR_LAB2BGR,
    )

    return {
        "bilateral": bilateral_image,
        "enhanced": maps["enhanced"],
        "white_top_hat": maps["white_top_hat"],
        "black_hat": maps["black_hat"],
    }


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# INDIVIDUAL COMPARISONS
# ============================================================

def save_two_panel_figure(
    left_image,
    right_image,
    left_title,
    right_title,
    figure_title,
    output_path,
):
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    figure.suptitle(
        figure_title,
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )
    axes[0].set_title(left_title, fontsize=16, pad=12)
    axes[1].set_title(right_title, fontsize=16, pad=12)
    axes[0].imshow(bgr_to_rgb(left_image))
    axes[1].imshow(bgr_to_rgb(right_image))

    for axis in axes:
        axis.axis("off")

    figure.subplots_adjust(
        top=0.82,
        bottom=0.04,
        left=0.03,
        right=0.97,
        wspace=0.08,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def save_individual_bilateral_comparison(
    class_name,
    original_labelled,
    bilateral_labelled,
):
    output_directory = BILATERAL_OUTPUT / class_name
    output_directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_directory / "01_original_labelled.png"),
        original_labelled,
    )
    cv2.imwrite(
        str(output_directory / "02_bilateral_labelled.png"),
        bilateral_labelled,
    )

    display_name = class_name.replace("_", " ").title()
    return save_two_panel_figure(
        original_labelled,
        bilateral_labelled,
        "Original PCB Image",
        "After Bilateral Filtering",
        f"Bilateral Filtering Analysis - {display_name}",
        output_directory / "bilateral_comparison.png",
    )


def save_individual_morphology_comparison(
    class_name,
    bilateral_labelled,
    enhanced_labelled,
    white_top_hat,
    black_hat,
):
    output_directory = MORPHOLOGY_OUTPUT / class_name
    output_directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_directory / "01_bilateral_labelled.png"),
        bilateral_labelled,
    )
    cv2.imwrite(
        str(output_directory / "02_bilateral_morphology_labelled.png"),
        enhanced_labelled,
    )
    cv2.imwrite(
        str(output_directory / "03_white_top_hat_map.png"),
        white_top_hat,
    )
    cv2.imwrite(
        str(output_directory / "04_black_hat_map.png"),
        black_hat,
    )

    display_name = class_name.replace("_", " ").title()
    return save_two_panel_figure(
        bilateral_labelled,
        enhanced_labelled,
        "After Bilateral Filtering",
        "After Bilateral + Multi-scale Morphology",
        f"Multi-scale Morphology Analysis - {display_name}",
        output_directory / "bilateral_morphology_comparison.png",
    )


# ============================================================
# COMBINED SIX-CLASS FIGURES
# ============================================================

def create_combined_figure(
    results,
    left_key,
    right_key,
    left_title,
    right_title,
    figure_title,
    output_path,
):
    figure, axes = plt.subplots(6, 2, figsize=(14, 26))
    figure.suptitle(
        figure_title,
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    axes[0, 0].set_title(left_title, fontsize=15, pad=12)
    axes[0, 1].set_title(right_title, fontsize=15, pad=12)

    for row_index, result in enumerate(results):
        axes[row_index, 0].imshow(
            bgr_to_rgb(result[left_key])
        )
        axes[row_index, 1].imshow(
            bgr_to_rgb(result[right_key])
        )

        defect_name = (
            result["class_name"].replace("_", " ").title()
        )
        axes[row_index, 0].set_ylabel(
            defect_name,
            fontsize=12,
            labelpad=15,
        )
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")

    figure.subplots_adjust(
        top=0.94,
        bottom=0.02,
        left=0.05,
        right=0.98,
        hspace=0.08,
        wspace=0.08,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("B1-M3-W2 VISUAL-EFFECT ANALYSIS")
    print("=" * 80)
    print(
        "Pipeline: Original -> Bilateral Filtering -> "
        "Multi-scale Top-Hat/Black-Hat Morphology"
    )

    check_inputs()
    results = []

    for class_id, class_name in CLASS_NAMES.items():
        print("\n" + "-" * 80)
        print(f"Processing: {class_name}")

        selected = select_representative_image(class_id)
        image_path = selected["image_path"]
        labels = selected["labels"]
        print(f"Selected image: {image_path.name}")

        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if original is None:
            raise RuntimeError(f"Unable to read:\n{image_path}")

        processed = process_image(original)
        original_labelled = draw_labels(
            original,
            labels,
            class_id,
        )
        bilateral_labelled = draw_labels(
            processed["bilateral"],
            labels,
            class_id,
        )
        enhanced_labelled = draw_labels(
            processed["enhanced"],
            labels,
            class_id,
        )

        bilateral_path = save_individual_bilateral_comparison(
            class_name,
            original_labelled,
            bilateral_labelled,
        )
        morphology_path = save_individual_morphology_comparison(
            class_name,
            bilateral_labelled,
            enhanced_labelled,
            processed["white_top_hat"],
            processed["black_hat"],
        )

        results.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "image_name": image_path.name,
                "original_labelled": original_labelled,
                "bilateral_labelled": bilateral_labelled,
                "enhanced_labelled": enhanced_labelled,
            }
        )

        print(f"Bilateral figure:\n{bilateral_path}")
        print(f"Morphology figure:\n{morphology_path}")

    print("\n" + "=" * 80)
    print("GENERATING COMBINED FIGURES")
    print("=" * 80)

    bilateral_combined_path = create_combined_figure(
        results,
        "original_labelled",
        "bilateral_labelled",
        "Original PCB Image",
        "After Bilateral Filtering",
        "Visual Effect of Bilateral Filtering Across Six PCB Defects",
        BILATERAL_OUTPUT / "bilateral_all_defects_comparison.png",
    )
    morphology_combined_path = create_combined_figure(
        results,
        "bilateral_labelled",
        "enhanced_labelled",
        "After Bilateral Filtering",
        "After Bilateral + Multi-scale Morphology",
        (
            "Visual Effect of Multi-scale Top-Hat/Black-Hat "
            "Morphology Across Six PCB Defects"
        ),
        (
            MORPHOLOGY_OUTPUT
            / "bilateral_morphology_all_defects_comparison.png"
        ),
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Combined bilateral figure:\n{bilateral_combined_path}")
    print(f"Combined morphology figure:\n{morphology_combined_path}")
    print(f"All outputs:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
