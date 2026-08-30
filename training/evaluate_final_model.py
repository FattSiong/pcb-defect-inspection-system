from pathlib import Path
import csv
import json

import torch
from ultralytics import YOLO


# =========================================================
# FINAL LOCKED CONFIGURATION: B1 + M3 + W2
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_morphology_tuning"
    / "stage3"
    / "seed_42"
    / "W2_train"
    / "weights"
    / "best.pt"
)

DATASET_ROOT = Path(
    r"C:\Users\hangg\Downloads\pcb_dataset_final_b1_m3_w2"
)
DATA_YAML = DATASET_ROOT / "data.yaml"
PROCESSING_CONFIG = DATASET_ROOT / "processing_config.json"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "final_evaluation"
    / "b1_m3_w2_seed_42"
)
EVALUATION_NAME = "test_evaluation"
PREDICTION_NAME = "test_prediction_samples"


# =========================================================
# FIXED TEST SETTINGS
# =========================================================

IMAGE_SIZE = 640
TEST_BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4

# These thresholds affect the saved prediction examples.
# They do not change the mAP calculation performed by val().
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

SAVE_PREDICTION_SAMPLES = True
PREDICTION_SAMPLE_COUNT = 50

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

FINAL_PROCESSING_CONFIGURATION = {
    "bilateral_d": 3,
    "sigma_color": 15,
    "sigma_space": 3,
    "kernel_sizes": [7, 15, 31],
    "alpha": 0.7,
    "beta": 0.7,
}


# =========================================================
# FILE AND ENVIRONMENT CHECKS
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
            "The final processing configuration was not found:\n"
            f"{PROCESSING_CONFIG}"
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

    if actual != FINAL_PROCESSING_CONFIGURATION:
        raise RuntimeError(
            "The final dataset does not use the locked B1 + M3 + W2 "
            "configuration.\n"
            f"Expected: {FINAL_PROCESSING_CONFIGURATION}\n"
            f"Actual  : {actual}"
        )

    test_statistics = next(
        (
            item
            for item in configuration.get("processing_statistics", [])
            if item.get("split") == "test"
        ),
        None,
    )

    if test_statistics is None:
        raise RuntimeError(
            "processing_config.json does not contain test-split "
            "processing statistics."
        )

    return float(test_statistics["seconds_per_image"])


def check_environment():
    print("=" * 80)
    print("FINAL B1 + M3 + W2 TEST EVALUATION")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final W2 model was not found:\n{MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "Final processed dataset was not found. Generate it "
            "with train, val and test splits before testing:\n"
            f"{DATA_YAML}"
        )

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory was not found:\n{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Test label directory was not found:\n{TEST_LABELS_DIR}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch and NVIDIA "
            "installation before running the final test."
        )

    test_images = list_images(TEST_IMAGES_DIR)
    test_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))

    if not test_images:
        raise RuntimeError(
            f"No test images were found in:\n{TEST_IMAGES_DIR}"
        )

    image_stems = {path.stem for path in test_images}
    label_stems = {path.stem for path in test_labels}
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels or missing_images:
        raise RuntimeError(
            "Test images and labels do not match. "
            f"Images without labels: {len(missing_labels)}; "
            f"labels without images: {len(missing_images)}."
        )

    yaml_text = DATA_YAML.read_text(encoding="utf-8")
    if "test:" not in yaml_text:
        raise RuntimeError(
            f"The dataset YAML does not define a test split:\n{DATA_YAML}"
        )

    preprocessing_seconds_per_image = (
        read_and_check_processing_configuration()
    )

    print(f"PyTorch        : {torch.__version__}")
    print(f"GPU            : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Model          : {MODEL_PATH}")
    print(f"Dataset        : {DATA_YAML}")
    print(f"Test images    : {len(test_images)}")
    print(f"Test labels    : {len(test_labels)}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Test batch     : {TEST_BATCH_SIZE}")
    print(f"Workers        : {WORKERS}")
    print(
        "Preprocess time: "
        f"{preprocessing_seconds_per_image * 1000:.3f} ms/image"
    )
    print(f"Output         : {OUTPUT_ROOT}")

    return test_images, preprocessing_seconds_per_image


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


def get_class_name(model, class_id):
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

    class_indices = getattr(
        metrics.box,
        "ap_class_index",
        None,
    )

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
        all_class_ids = sorted(int(key) for key in model.names)
    else:
        all_class_ids = list(range(len(model.names)))

    class_results = []

    for class_id in all_class_ids:
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

        # Ultralytics returns maps indexed by the original class ID.
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

        class_results.append(
            {
                "class_id": class_id,
                "class_name": get_class_name(model, class_id),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "map50": ap50,
                "map50_95": map50_95,
            }
        )

    return class_results


def extract_test_results(
    metrics,
    model,
    test_image_count,
    preprocessing_seconds_per_image,
):
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
    preprocessing_ms_per_image = (
        preprocessing_seconds_per_image * 1000.0
    )
    end_to_end_ms_per_image = (
        preprocessing_ms_per_image + yolo_total_ms_per_image
    )
    frames_per_second = (
        1000.0 / end_to_end_ms_per_image
        if end_to_end_ms_per_image > 0
        else 0.0
    )

    per_class = extract_per_class_metrics(metrics, model)
    lowest_class_map50_95 = min(
        result["map50_95"] for result in per_class
    )

    return {
        "method": "Bilateral + Multi-scale Top-Hat/Black-Hat",
        "configuration": "B1_M3_W2",
        "model": "YOLOv8n",
        "model_path": str(MODEL_PATH),
        "dataset": str(DATA_YAML),
        "split": "test",
        "test_images": test_image_count,
        "image_size": IMAGE_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "processing_configuration": FINAL_PROCESSING_CONFIGURATION,
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
        "preprocessing_ms_per_image": preprocessing_ms_per_image,
        "yolo_total_ms_per_image": yolo_total_ms_per_image,
        "end_to_end_ms_per_image": end_to_end_ms_per_image,
        "frames_per_second": frames_per_second,
    }


# =========================================================
# RESULT FILES
# =========================================================

def save_results(results, result_directory):
    result_directory.mkdir(parents=True, exist_ok=True)

    json_path = result_directory / "test_results.json"
    json_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    overall_path = result_directory / "test_metrics_summary.csv"
    overall = results["overall_metrics"]

    with overall_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "configuration",
                "test_images",
                "precision",
                "recall",
                "f1_score",
                "map50",
                "map50_95",
                "lowest_class_map50_95",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": results["method"],
                "configuration": results["configuration"],
                "test_images": results["test_images"],
                **overall,
            }
        )

    class_path = result_directory / "test_metrics_by_class.csv"
    with class_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "map50",
                "map50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(results["per_class_metrics"])

    speed_path = result_directory / "test_processing_speed.csv"
    with speed_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "milliseconds_per_image"])

        writer.writerow(
            ["image_enhancement", results["preprocessing_ms_per_image"]]
        )

        for stage, value in results["yolo_speed_ms_per_image"].items():
            writer.writerow([f"yolo_{stage}", value])

        writer.writerow(
            ["yolo_total", results["yolo_total_ms_per_image"]]
        )
        writer.writerow(
            ["end_to_end_total", results["end_to_end_ms_per_image"]]
        )
        writer.writerow(["frames_per_second", results["frames_per_second"]])

    print("\nNumerical results saved:")
    print(f"  {json_path}")
    print(f"  {overall_path}")
    print(f"  {class_path}")
    print(f"  {speed_path}")


# =========================================================
# TEST EVALUATION AND PREDICTION EXAMPLES
# =========================================================

def run_test_evaluation(
    model,
    test_image_count,
    preprocessing_seconds_per_image,
):
    print("\n" + "=" * 80)
    print("EVALUATING LOCKED MODEL ON THE TEST SPLIT")
    print("=" * 80)

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
    )

    result_directory = Path(metrics.save_dir)
    results = extract_test_results(
        metrics,
        model,
        test_image_count,
        preprocessing_seconds_per_image,
    )
    save_results(results, result_directory)

    overall = results["overall_metrics"]

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
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
    print(
        "End-to-end speed: "
        f"{results['end_to_end_ms_per_image']:.3f} ms/image"
    )
    print(
        "Estimated FPS   : "
        f"{results['frames_per_second']:.3f}"
    )

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
    prediction_directory = OUTPUT_ROOT / PREDICTION_NAME

    print(f"Saved prediction examples: {saved_count}")
    print(f"Prediction directory:\n{prediction_directory}")


# =========================================================
# MAIN
# =========================================================

def main():
    (
        test_images,
        preprocessing_seconds_per_image,
    ) = check_environment()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    result_directory = run_test_evaluation(
        model,
        len(test_images),
        preprocessing_seconds_per_image,
    )
    save_prediction_samples(model, test_images)

    print("\n" + "=" * 80)
    print("FINAL TESTING COMPLETED")
    print("=" * 80)
    print(f"Evaluation results:\n{result_directory}")


if __name__ == "__main__":
    main()
