"""Evaluate the locked Gaussian-LBP W1 model on the test split."""

from pathlib import Path
import csv
import json
import math

import torch
from ultralytics import YOLO


# =========================================================
# FINAL LOCKED CONFIGURATION: G2 + L1 + W1
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT /
    "runs" /
    "gaussian_lbp" /
    "gaussian_lbp_final_w1" /
    "weights" /
    "best.pt"
)

# Change this setting only if the final dataset has a different location.
DATASET_ROOT = (
    Path.home() /
    "Downloads" /
    "pcb_dataset_gaussian_lbp_final_w1"
)

DATA_YAML = DATASET_ROOT / "data.yaml"
PROCESSING_CONFIG = DATASET_ROOT / "processing_config.json"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "final_evaluation" /
    "gaussian_lbp_final_w1_seed_42"
)
EVALUATION_NAME = "test_evaluation"
PREDICTION_NAME = "test_prediction_samples"


# =========================================================
# FIXED TEST SETTINGS
# =========================================================

IMAGE_SIZE = 640
TEST_BATCH_SIZE = 4
DEVICE = 0

# These thresholds affect saved prediction examples only. They do not
# change the mAP calculation performed by model.val().
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
    "gaussian_kernel": 3,
    "gaussian_sigma": 0.8,
    "lbp_points": 8,
    "lbp_radius": 1.0,
    "lbp_weight": 0.15,
    "gaussian_weight": 0.85,
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


def read_and_check_processing_configuration():
    if not PROCESSING_CONFIG.exists():
        raise FileNotFoundError(
            "Final processing configuration not found:\n"
            f"{PROCESSING_CONFIG}"
        )

    configuration = json.loads(
        PROCESSING_CONFIG.read_text(encoding="utf-8")
    )

    if configuration.get("method") != (
        "gaussian_uniform_lbp_luminance_fusion"
    ):
        raise RuntimeError(
            "The final dataset is not a Gaussian-LBP fusion dataset."
        )

    generated_splits = set(configuration.get("splits", []))

    if not {"train", "val", "test"}.issubset(generated_splits):
        raise RuntimeError(
            "The final dataset configuration must contain train, "
            "val and test splits."
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

    for parameter, expected in (
        FINAL_PROCESSING_CONFIGURATION.items()
    ):
        actual_value = actual[parameter]

        if not values_match(actual_value, expected):
            raise RuntimeError(
                f"Final {parameter} mismatch: expected {expected}, "
                f"found {actual_value}."
            )

    test_statistics = next(
        (
            item
            for item in configuration.get(
                "processing_statistics",
                [],
            )
            if item.get("split") == "test"
        ),
        None,
    )

    if test_statistics is None:
        raise RuntimeError(
            "processing_config.json does not contain test-split "
            "processing statistics."
        )

    return (
        configuration,
        float(test_statistics["seconds_per_image"]),
    )


def check_environment():
    print("=" * 80)
    print("FINAL GAUSSIAN + SINGLE-SCALE LBP TEST EVALUATION")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Final W1 model not found:\n"
            f"{MODEL_PATH}\n"
            "Check MODEL_PATH at the top of this file."
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "Final dataset YAML not found:\n"
            f"{DATA_YAML}\n"
            "Check DATASET_ROOT at the top of this file."
        )

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Test label directory not found:\n{TEST_LABELS_DIR}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch and NVIDIA setup."
        )

    test_images = list_images(TEST_IMAGES_DIR)
    test_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))

    if not test_images:
        raise RuntimeError(
            f"No test images found in:\n{TEST_IMAGES_DIR}"
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
            f"The dataset YAML has no test split:\n{DATA_YAML}"
        )

    (
        processing_configuration,
        generation_seconds_per_image,
    ) = read_and_check_processing_configuration()

    print(f"PyTorch         : {torch.__version__}")
    print(f"GPU             : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Model           : {MODEL_PATH}")
    print(f"Dataset         : {DATA_YAML}")
    print(f"Test images     : {len(test_images)}")
    print(f"Test labels     : {len(test_labels)}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Test batch      : {TEST_BATCH_SIZE}")
    print(
        "Dataset generation time: "
        f"{generation_seconds_per_image * 1000.0:.3f} ms/image"
    )
    print(f"Output          : {OUTPUT_ROOT}")

    return (
        test_images,
        processing_configuration,
        generation_seconds_per_image,
    )


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
        map50 = (
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

        class_results.append(
            {
                "class_id": class_id,
                "class_name": get_class_name(model, class_id),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "map50": map50,
                "map50_95": map50_95,
            }
        )

    return class_results


def extract_test_results(
    metrics,
    model,
    test_image_count,
    processing_configuration,
    generation_seconds_per_image,
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

    yolo_speed = {
        stage: float(milliseconds)
        for stage, milliseconds in metrics.speed.items()
    }
    yolo_total_ms_per_image = sum(yolo_speed.values())
    generation_ms_per_image = (
        generation_seconds_per_image * 1000.0
    )
    estimated_pipeline_ms_per_image = (
        generation_ms_per_image + yolo_total_ms_per_image
    )
    estimated_pipeline_fps = (
        1000.0 / estimated_pipeline_ms_per_image
        if estimated_pipeline_ms_per_image > 0
        else 0.0
    )

    per_class = extract_per_class_metrics(metrics, model)
    lowest_class_map50_95 = min(
        result["map50_95"] for result in per_class
    )

    return {
        "method": "Gaussian + single-scale uniform LBP fusion",
        "configuration": "G2_L1_W1",
        "model": "YOLOv8n",
        "model_path": str(MODEL_PATH),
        "dataset": str(DATA_YAML),
        "split": "test",
        "test_images": test_image_count,
        "image_size": IMAGE_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "device": DEVICE,
        "processing_configuration": processing_configuration,
        "locked_parameters": FINAL_PROCESSING_CONFIGURATION,
        "overall_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "map50": map50,
            "map50_95": map50_95,
            "lowest_class_map50_95": lowest_class_map50_95,
        },
        "per_class_metrics": per_class,
        "yolo_speed_ms_per_image": yolo_speed,
        "yolo_total_ms_per_image": yolo_total_ms_per_image,
        "offline_dataset_generation_ms_per_image": (
            generation_ms_per_image
        ),
        "estimated_pipeline_ms_per_image": (
            estimated_pipeline_ms_per_image
        ),
        "estimated_pipeline_fps": estimated_pipeline_fps,
        "speed_note": (
            "Offline dataset-generation time includes image and label I/O; "
            "it is not a pure in-memory preprocessing benchmark."
        ),
    }


# =========================================================
# REPORT FILES
# =========================================================

def save_results(results, result_directory):
    result_directory.mkdir(parents=True, exist_ok=True)

    json_path = result_directory / "test_results.json"
    json_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    overall = results["overall_metrics"]
    overall_path = result_directory / "test_metrics_summary.csv"

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
            [
                "offline_dataset_generation",
                results["offline_dataset_generation_ms_per_image"],
            ]
        )

        for stage, value in results[
            "yolo_speed_ms_per_image"
        ].items():
            writer.writerow([f"yolo_{stage}", value])

        writer.writerow(
            ["yolo_total", results["yolo_total_ms_per_image"]]
        )
        writer.writerow(
            [
                "estimated_pipeline_total",
                results["estimated_pipeline_ms_per_image"],
            ]
        )
        writer.writerow(
            ["estimated_pipeline_fps", results["estimated_pipeline_fps"]]
        )

    print()
    print("Reports saved:")
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
    processing_configuration,
    generation_seconds_per_image,
):
    print()
    print("=" * 80)
    print("EVALUATING LOCKED W1 MODEL ON THE TEST SPLIT")
    print("=" * 80)

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=TEST_BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_ROOT),
        name=EVALUATION_NAME,
        exist_ok=True,
        plots=True,
    )

    result_directory = Path(metrics.save_dir)
    results = extract_test_results(
        metrics=metrics,
        model=model,
        test_image_count=test_image_count,
        processing_configuration=processing_configuration,
        generation_seconds_per_image=generation_seconds_per_image,
    )
    save_results(results, result_directory)

    overall = results["overall_metrics"]

    print()
    print("=" * 80)
    print("FINAL GAUSSIAN-LBP TEST RESULTS")
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
        "Estimated pipeline speed: "
        f"{results['estimated_pipeline_ms_per_image']:.3f} ms/image"
    )
    print(
        "Estimated pipeline FPS  : "
        f"{results['estimated_pipeline_fps']:.3f}"
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

    print()
    print("=" * 80)
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
        processing_configuration,
        generation_seconds_per_image,
    ) = check_environment()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    result_directory = run_test_evaluation(
        model=model,
        test_image_count=len(test_images),
        processing_configuration=processing_configuration,
        generation_seconds_per_image=generation_seconds_per_image,
    )
    save_prediction_samples(model, test_images)

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Evaluation reports:\n{result_directory}")


if __name__ == "__main__":
    main()

