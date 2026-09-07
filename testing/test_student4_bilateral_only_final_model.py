"""Evaluate the trained bilateral-only B1 model on the test split."""

from pathlib import Path
import csv
import json
import time

import torch
from ultralytics import YOLO


# =========================================================
# PATHS AND LOCKED CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_only"
    / "seed_42"
    / "bilateral_only_b1_train"
    / "weights"
    / "best.pt"
)

DATASET_ROOT = Path(
    r"C:\Users\hangg\Downloads\pcb_dataset_bilateral_only_b1"
)
DATA_YAML = DATASET_ROOT / "data.yaml"
PROCESSING_CONFIG = DATASET_ROOT / "processing_config.json"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "final_evaluation"
    / "bilateral_only_b1_seed_42"
)
EVALUATION_NAME = "test_evaluation"
PREDICTION_NAME = "test_prediction_samples"

EXPECTED_PROCESSING = {
    "bilateral_d": 3,
    "sigma_color": 15.0,
    "sigma_space": 3.0,
    "kernel_sizes": [7, 15, 31],
    "alpha": 0.0,
    "beta": 0.0,
}


# =========================================================
# FIXED TEST SETTINGS
# =========================================================

IMAGE_SIZE = 640
TEST_BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4

SAVE_PREDICTION_SAMPLES = True
PREDICTION_SAMPLE_COUNT = 50
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

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
# CHECK MODEL AND TEST DATASET
# =========================================================

def list_images(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_and_check_processing_configuration():
    if not PROCESSING_CONFIG.exists():
        raise FileNotFoundError(
            f"Processing configuration not found:\n{PROCESSING_CONFIG}"
        )

    configuration = json.loads(
        PROCESSING_CONFIG.read_text(encoding="utf-8")
    )
    bilateral = configuration.get("bilateral_filter", {})
    morphology = configuration.get("morphology", {})

    actual = {
        "bilateral_d": bilateral.get("diameter"),
        "sigma_color": bilateral.get("sigma_color"),
        "sigma_space": bilateral.get("sigma_space"),
        "kernel_sizes": morphology.get("kernel_sizes"),
        "alpha": morphology.get("alpha"),
        "beta": morphology.get("beta"),
    }

    if actual != EXPECTED_PROCESSING:
        raise RuntimeError(
            "The selected dataset is not the locked bilateral-only B1 "
            "dataset.\n"
            f"Expected: {EXPECTED_PROCESSING}\n"
            f"Actual  : {actual}"
        )

    if "test" not in configuration.get("splits", []):
        raise RuntimeError(
            "processing_config.json does not include the test split."
        )

    return actual


def check_environment():
    print("=" * 80)
    print("BILATERAL-ONLY B1 FINAL TEST")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Bilateral-only best.pt not found:\n{MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not TEST_IMAGES_DIR.exists() or not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            "The bilateral-only test image or label directory is missing."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processing = read_and_check_processing_configuration()
    test_images = list_images(TEST_IMAGES_DIR)
    test_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))

    image_stems = {path.stem for path in test_images}
    label_stems = {path.stem for path in test_labels}

    if not image_stems:
        raise RuntimeError("No test images were found.")

    if image_stems != label_stems:
        raise RuntimeError(
            "Test images and labels do not match. "
            f"Images without labels: {len(image_stems - label_stems)}; "
            f"labels without images: {len(label_stems - image_stems)}."
        )

    if "test:" not in DATA_YAML.read_text(encoding="utf-8"):
        raise RuntimeError("data.yaml does not define a test split.")

    print(f"GPU            : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Model          : {MODEL_PATH}")
    print(f"Dataset        : {DATA_YAML}")
    print(f"Test images    : {len(test_images)}")
    print(f"Test labels    : {len(test_labels)}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Test batch     : {TEST_BATCH_SIZE}")
    print(f"Workers        : {WORKERS}")
    print(f"Processing     : {processing}")
    print(f"Output         : {OUTPUT_ROOT}")

    return test_images


# =========================================================
# EXTRACT METRICS
# =========================================================

def to_float_list(values):
    if values is None:
        return []

    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, list):
        values = list(values)

    return [float(value) for value in values]


def class_name(model, class_id):
    if isinstance(model.names, dict):
        return model.names.get(class_id, f"class_{class_id}")

    if class_id < len(model.names):
        return model.names[class_id]

    return f"class_{class_id}"


def extract_per_class_metrics(metrics, model):
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

    if isinstance(model.names, dict):
        class_ids = sorted(int(key) for key in model.names)
    else:
        class_ids = list(range(len(model.names)))

    results = []

    for class_id in class_ids:
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

        results.append(
            {
                "class_id": class_id,
                "class_name": class_name(model, class_id),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "map50": ap50,
                "map50_95": map50_95,
            }
        )

    return results


def extract_results(metrics, model, test_count, wall_time_seconds):
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    speed = {
        stage: float(milliseconds)
        for stage, milliseconds in metrics.speed.items()
    }
    yolo_total_ms_per_image = sum(speed.values())
    per_class = extract_per_class_metrics(metrics, model)
    lowest_class_map50_95 = min(
        result["map50_95"] for result in per_class
    )

    return {
        "method": "Bilateral Filtering Only",
        "configuration": "bilateral_only_b1",
        "model": "YOLOv8n",
        "model_path": str(MODEL_PATH),
        "dataset": str(DATA_YAML),
        "split": "test",
        "test_images": test_count,
        "image_size": IMAGE_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "processing_configuration": EXPECTED_PROCESSING,
        "overall_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "map50": map50,
            "map50_95": map50_95,
            "lowest_class_map50_95": lowest_class_map50_95,
        },
        "per_class_metrics": per_class,
        "yolo_speed_ms_per_image": speed,
        "yolo_total_ms_per_image": yolo_total_ms_per_image,
        "total_test_wall_time_seconds": wall_time_seconds,
    }


# =========================================================
# SAVE RESULT FILES
# =========================================================

def save_results(results, result_directory):
    result_directory.mkdir(parents=True, exist_ok=True)

    json_path = result_directory / "test_results.json"
    json_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    overall_path = result_directory / "test_metrics_summary.csv"
    with overall_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "method",
            "configuration",
            "test_images",
            "precision",
            "recall",
            "f1_score",
            "map50",
            "map50_95",
            "lowest_class_map50_95",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "method": results["method"],
                "configuration": results["configuration"],
                "test_images": results["test_images"],
                **results["overall_metrics"],
            }
        )

    class_path = result_directory / "test_metrics_by_class.csv"
    with class_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1_score",
            "map50",
            "map50_95",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results["per_class_metrics"])

    speed_path = result_directory / "test_processing_speed.csv"
    with speed_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "value"])

        for stage, value in results["yolo_speed_ms_per_image"].items():
            writer.writerow([f"yolo_{stage}_ms_per_image", value])

        writer.writerow(
            ["yolo_total_ms_per_image", results["yolo_total_ms_per_image"]]
        )
        writer.writerow(
            [
                "total_test_wall_time_seconds",
                results["total_test_wall_time_seconds"],
            ]
        )

    print("\nSaved numerical results:")
    print(f"  {json_path}")
    print(f"  {overall_path}")
    print(f"  {class_path}")
    print(f"  {speed_path}")


# =========================================================
# TEST AND PREDICTION EXAMPLES
# =========================================================

def run_test(model, test_count):
    print("\n" + "=" * 80)
    print("EVALUATING BILATERAL-ONLY BEST.PT ON TEST")
    print("=" * 80)

    start_time = time.perf_counter()
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=TEST_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUTPUT_ROOT),
        name=EVALUATION_NAME,
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    wall_time_seconds = time.perf_counter() - start_time

    result_directory = Path(metrics.save_dir)
    results = extract_results(
        metrics,
        model,
        test_count,
        wall_time_seconds,
    )
    save_results(results, result_directory)

    overall = results["overall_metrics"]

    print("\n" + "=" * 80)
    print("BILATERAL-ONLY FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Precision       : {overall['precision']:.6f}")
    print(f"Recall          : {overall['recall']:.6f}")
    print(f"F1-score        : {overall['f1_score']:.6f}")
    print(f"mAP50           : {overall['map50']:.6f}")
    print(f"mAP50-95        : {overall['map50_95']:.6f}")
    print(
        "Lowest class AP : "
        f"{overall['lowest_class_map50_95']:.6f}"
    )
    print(
        "YOLO speed      : "
        f"{results['yolo_total_ms_per_image']:.3f} ms/image"
    )
    print(f"Wall time       : {wall_time_seconds:.3f} seconds")

    return result_directory


def select_prediction_samples(test_images):
    sample_count = min(PREDICTION_SAMPLE_COUNT, len(test_images))

    if sample_count == len(test_images):
        return test_images

    if sample_count == 1:
        return [test_images[0]]

    indices = {
        round(index * (len(test_images) - 1) / (sample_count - 1))
        for index in range(sample_count)
    }
    return [test_images[index] for index in sorted(indices)]


def save_prediction_samples(model, test_images):
    if not SAVE_PREDICTION_SAMPLES:
        return

    samples = select_prediction_samples(test_images)

    print("\n" + "=" * 80)
    print(f"SAVING {len(samples)} TEST PREDICTION EXAMPLES")
    print("=" * 80)

    predictions = model.predict(
        source=[str(path) for path in samples],
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        project=str(OUTPUT_ROOT),
        name=PREDICTION_NAME,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        stream=True,
        verbose=False,
    )

    saved_count = sum(1 for _ in predictions)
    print(f"Saved prediction examples: {saved_count}")
    print(f"Prediction directory:\n{OUTPUT_ROOT / PREDICTION_NAME}")


def main():
    test_images = check_environment()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    result_directory = run_test(model, len(test_images))
    save_prediction_samples(model, test_images)

    print("\n" + "=" * 80)
    print("BILATERAL-ONLY FINAL TEST COMPLETED")
    print("=" * 80)
    print(f"Evaluation results:\n{result_directory}")


if __name__ == "__main__":
    main()
