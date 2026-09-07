"""Test the final G2-L1-W1 Gaussian-LBP model with report outputs."""

from pathlib import Path
import json
import math
import time

import pandas as pd
from ultralytics import YOLO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "gaussian_lbp"
    / "gaussian_lbp_final_w1"
    / "weights"
    / "best.pt"
)

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "pcb_dataset_gaussian_lbp_final_w1"
)
DATA_YAML = DATASET_ROOT / "data.yaml"
PROCESSING_CONFIG = DATASET_ROOT / "processing_config.json"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "gaussian_lbp_final_test"
    / "fusion_test"
)
PER_CLASS_DIR = OUTPUT_ROOT / "per_class"
GRAPH_DIR = OUTPUT_ROOT / "graphs"
YOLO_RESULT_NAME = "yolo_results"

METHOD_NAME = "Gaussian Filtering + Uniform LBP Luminance Fusion"
CONFIGURATION_NAME = "G2-L1-W1"

EXPECTED_PROCESSING = {
    "gaussian_kernel": 3,
    "gaussian_sigma": 0.8,
    "lbp_points": 8,
    "lbp_radius": 1.0,
    "lbp_weight": 0.15,
    "gaussian_weight": 0.85,
}

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# =========================================================
# SAME TEST SETTINGS AS test_student1_final_model.py
# =========================================================

IMAGE_SIZE = 640
TEST_BATCH_SIZE = 4
DEVICE = 0
WORKERS = 0

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


# =========================================================
# INPUT VALIDATION
# =========================================================

def list_images(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def values_match(actual, expected):
    if actual is None:
        return False

    if isinstance(expected, float):
        return math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    return actual == expected


def validate_processing_configuration():
    if not PROCESSING_CONFIG.exists():
        raise FileNotFoundError(
            f"Processing configuration not found:\n{PROCESSING_CONFIG}"
        )

    configuration = json.loads(
        PROCESSING_CONFIG.read_text(encoding="utf-8")
    )

    if configuration.get("method") != (
        "gaussian_uniform_lbp_luminance_fusion"
    ):
        raise RuntimeError(
            "The selected dataset is not a Gaussian-LBP fusion dataset."
        )

    gaussian = configuration.get("gaussian_filter", {})
    lbp = configuration.get("lbp", {})
    fusion = configuration.get("fusion", {})

    actual = {
        "gaussian_kernel": gaussian.get("kernel_size"),
        "gaussian_sigma": gaussian.get("sigma"),
        "lbp_points": lbp.get("points"),
        "lbp_radius": lbp.get("radius"),
        "lbp_weight": fusion.get("lbp_weight"),
        "gaussian_weight": fusion.get("gaussian_weight"),
    }

    for parameter, expected in EXPECTED_PROCESSING.items():
        if not values_match(actual.get(parameter), expected):
            raise RuntimeError(
                f"Final {parameter} mismatch: expected {expected}, "
                f"found {actual.get(parameter)}."
            )

    if "test" not in configuration.get("splits", []):
        raise RuntimeError(
            "processing_config.json does not include the test split."
        )

    return actual


def validate_inputs():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Gaussian-LBP best.pt not found:\n{MODEL_PATH}\n"
            "Update MODEL_PATH if the trained model was saved elsewhere."
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Gaussian-LBP dataset YAML not found:\n{DATA_YAML}\n"
            "Update DATASET_ROOT if the final dataset is elsewhere."
        )

    if not TEST_IMAGES_DIR.exists() or not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            "The Gaussian-LBP test image or label directory is missing."
        )

    processing = validate_processing_configuration()
    test_images = list_images(TEST_IMAGES_DIR)
    test_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))
    image_stems = {path.stem for path in test_images}
    label_stems = {path.stem for path in test_labels}

    if not image_stems:
        raise RuntimeError("No Gaussian-LBP test images were found.")

    if image_stems != label_stems:
        raise RuntimeError(
            "Test images and labels do not match. "
            f"Images without labels: {len(image_stems - label_stems)}; "
            f"labels without images: {len(label_stems - image_stems)}."
        )

    if "test:" not in DATA_YAML.read_text(encoding="utf-8"):
        raise RuntimeError("data.yaml does not define a test split.")

    return processing, len(test_images)


# =========================================================
# METRIC EXTRACTION
# =========================================================

def to_float_list(values):
    if values is None:
        return []

    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, list):
        values = list(values)

    return [float(value) for value in values]


def extract_per_class_rows(metrics):
    precision_values = to_float_list(metrics.box.p)
    recall_values = to_float_list(metrics.box.r)
    ap50_values = to_float_list(metrics.box.ap50)
    map50_95_values = to_float_list(metrics.box.maps)

    class_indices = getattr(metrics.box, "ap_class_index", None)

    if class_indices is None:
        class_indices = list(range(len(precision_values)))
    elif hasattr(class_indices, "tolist"):
        class_indices = class_indices.tolist()
    else:
        class_indices = list(class_indices)

    row_for_class = {
        int(class_id): row
        for row, class_id in enumerate(class_indices)
    }
    rows = []

    for class_id, class_name in CLASS_NAMES.items():
        row = row_for_class.get(class_id)
        precision = (
            precision_values[row]
            if row is not None and row < len(precision_values)
            else 0.0
        )
        recall = (
            recall_values[row]
            if row is not None and row < len(recall_values)
            else 0.0
        )
        ap50 = (
            ap50_values[row]
            if row is not None and row < len(ap50_values)
            else 0.0
        )
        map50_95 = (
            map50_95_values[class_id]
            if class_id < len(map50_95_values)
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        rows.append(
            {
                "Class ID": class_id,
                "Defect Class": class_name,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1_score,
                "mAP50": ap50,
                "mAP50-95": map50_95,
            }
        )

    return rows


# =========================================================
# REPORT GRAPHS
# =========================================================

def add_bar_values(bars, values):
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )


def generate_graphs(
    per_class_df,
    precision,
    recall,
    f1_score,
    map50,
    map50_95,
):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
    ]
    metric_values = [precision, recall, f1_score, map50, map50_95]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(metric_names, metric_values)
    plt.title(f"Overall Test Performance of {CONFIGURATION_NAME}")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    add_bar_values(bars, metric_values)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "overall_test_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    values = per_class_df["mAP50-95"]
    bars = plt.bar(per_class_df["Defect Class"], values)
    plt.title(f"Per-Class Test mAP50-95 of {CONFIGURATION_NAME}")
    plt.xlabel("PCB Defect Class")
    plt.ylabel("mAP50-95")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    add_bar_values(bars, values)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "per_class_map50_95.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    metrics_to_plot = [
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
    ]
    (
        per_class_df.set_index("Defect Class")[metrics_to_plot]
        .plot(kind="bar", figsize=(12, 7))
    )
    plt.title(f"Per-Class Test Performance of {CONFIGURATION_NAME}")
    plt.xlabel("PCB Defect Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "per_class_all_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(overall_data, per_class_rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PER_CLASS_DIR.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([overall_data])
    overall_csv = OUTPUT_ROOT / "overall_test_metrics.csv"
    overall_json = OUTPUT_ROOT / "overall_test_metrics.json"
    overall_df.to_csv(overall_csv, index=False)
    overall_json.write_text(
        json.dumps(overall_data, indent=4),
        encoding="utf-8",
    )

    per_class_df = pd.DataFrame(per_class_rows)
    all_classes_csv = OUTPUT_ROOT / "all_classes_summary.csv"
    per_class_df.to_csv(all_classes_csv, index=False)

    for row in per_class_rows:
        class_path = PER_CLASS_DIR / f"{row['Defect Class']}.csv"
        pd.DataFrame([row]).to_csv(class_path, index=False)

    return overall_csv, overall_json, all_classes_csv, per_class_df


# =========================================================
# FINAL TEST
# =========================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PER_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    processing, test_image_count = validate_inputs()

    print("=" * 80)
    print("GAUSSIAN-LBP FINAL TEST")
    print("=" * 80)
    print(f"Model            : {MODEL_PATH}")
    print(f"Dataset          : {DATA_YAML}")
    print(f"Test images      : {test_image_count}")
    print(f"Configuration    : {CONFIGURATION_NAME}")
    print(f"Gaussian kernel  : {processing['gaussian_kernel']}")
    print(f"Gaussian sigma   : {processing['gaussian_sigma']}")
    print(f"LBP points       : {processing['lbp_points']}")
    print(f"LBP radius       : {processing['lbp_radius']}")
    print(f"LBP weight       : {processing['lbp_weight']}")
    print(f"Gaussian weight  : {processing['gaussian_weight']}")
    print("YOLO model       : YOLOv8n")
    print(f"Image size       : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Test batch size  : {TEST_BATCH_SIZE}")
    print(f"Workers          : {WORKERS}")

    model = YOLO(str(MODEL_PATH))

    print("\nRunning Gaussian-LBP final test...")
    start_time = time.perf_counter()
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=TEST_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        save_json=False,
        project=str(OUTPUT_ROOT),
        name=YOLO_RESULT_NAME,
        exist_ok=True,
    )
    total_test_time = time.perf_counter() - start_time

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    speed = metrics.speed
    yolo_preprocess = float(speed.get("preprocess", 0.0))
    inference_time = float(speed.get("inference", 0.0))
    postprocess_time = float(speed.get("postprocess", 0.0))
    total_yolo_time = (
        yolo_preprocess + inference_time + postprocess_time
    )

    overall_data = {
        "Technique": METHOD_NAME,
        "Configuration": CONFIGURATION_NAME,
        "Model": "YOLOv8n",
        "Gaussian Kernel": processing["gaussian_kernel"],
        "Gaussian Sigma": processing["gaussian_sigma"],
        "LBP Points": processing["lbp_points"],
        "LBP Radius": processing["lbp_radius"],
        "LBP Weight": processing["lbp_weight"],
        "Gaussian Weight": processing["gaussian_weight"],
        "Test Images": test_image_count,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "YOLO Preprocess Time (ms/image)": yolo_preprocess,
        "Inference Time (ms/image)": inference_time,
        "Postprocess Time (ms/image)": postprocess_time,
        "Total YOLO Time (ms/image)": total_yolo_time,
        "Total Test Wall Time (seconds)": total_test_time,
    }

    per_class_rows = extract_per_class_rows(metrics)
    (
        overall_csv,
        overall_json,
        all_classes_csv,
        per_class_df,
    ) = save_outputs(overall_data, per_class_rows)

    generate_graphs(
        per_class_df,
        precision,
        recall,
        f1_score,
        map50,
        map50_95,
    )

    print("\n" + "=" * 80)
    print("OVERALL GAUSSIAN-LBP TEST RESULT")
    print("=" * 80)
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1_score:.4f}")
    print(f"mAP50        : {map50:.4f}")
    print(f"mAP50-95     : {map50_95:.4f}")
    print("\nYOLO Speed")
    print(f"Preprocess   : {yolo_preprocess:.3f} ms/image")
    print(f"Inference    : {inference_time:.3f} ms/image")
    print(f"Postprocess  : {postprocess_time:.3f} ms/image")
    print(f"Total YOLO   : {total_yolo_time:.3f} ms/image")
    print(f"Wall Time    : {total_test_time:.2f} seconds")

    display_columns = [
        "Defect Class",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
    ]
    print("\n" + "=" * 80)
    print("PER-CLASS TEST RESULTS")
    print("=" * 80)
    print(
        per_class_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print(f"Overall CSV : {overall_csv}")
    print(f"Overall JSON: {overall_json}")
    print(f"All classes : {all_classes_csv}")
    print(f"Per class   : {PER_CLASS_DIR}")
    print(f"Graphs      : {GRAPH_DIR}")
    print(f"YOLO plots  : {OUTPUT_ROOT / YOLO_RESULT_NAME}")

    print("\nGAUSSIAN-LBP FINAL TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
