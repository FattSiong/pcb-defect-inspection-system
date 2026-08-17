from pathlib import Path
import csv
import torch

from ultralytics import YOLO


# =========================================================
# Experiment configuration
# =========================================================

BASELINE_DATA = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_baseline\data.yaml"
)

CLAHE_DATA = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_clahe\data.yaml"
)

MODEL_NAME = "yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = -1
DEVICE = 0
SEED = 42

PROJECT_DIR = "runs/pcb_comparison"


# =========================================================
# Check environment
# =========================================================

def check_environment():

    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Please check the PyTorch GPU installation."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print()


# =========================================================
# Train one experiment
# =========================================================

def train_experiment(
    experiment_name,
    data_path
):

    print("\n" + "=" * 60)
    print(f"Training: {experiment_name}")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}"
        )

    # Load the SAME pretrained model
    # for every experiment
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(data_path),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        seed=SEED,
        deterministic=True,

        project=PROJECT_DIR,
        name=experiment_name,

        save=True,
        plots=True,
        val=True,

        exist_ok=True
    )

    best_model_path = (
        Path(PROJECT_DIR)
        / experiment_name
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found: "
            f"{best_model_path}"
        )

    print(
        f"\nBest model saved at: "
        f"{best_model_path}"
    )

    return best_model_path


# =========================================================
# Evaluate model using TEST set
# =========================================================

def evaluate_experiment(
    experiment_name,
    best_model_path,
    data_path
):

    print("\n" + "=" * 60)
    print(f"Testing: {experiment_name}")
    print("=" * 60)

    model = YOLO(
        str(best_model_path)
    )

    metrics = model.val(
        data=str(data_path),

        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        project=PROJECT_DIR,
        name=f"{experiment_name}_test",

        plots=True,
        exist_ok=True
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    results = {
        "experiment": experiment_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "map50": map50,
        "map50_95": map50_95
    }

    print("\nFinal TEST Results")
    print("-" * 40)

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"mAP50     : {map50:.4f}"
    )

    print(
        f"mAP50-95  : {map50_95:.4f}"
    )

    return results


# =========================================================
# Save comparison to CSV
# =========================================================

def save_results(results):

    output_path = (
        Path(PROJECT_DIR)
        / "comparison_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment",
        "precision",
        "recall",
        "f1_score",
        "map50",
        "map50_95"
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nComparison results saved to:"
        f"\n{output_path}"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    check_environment()

    all_results = []

    # -----------------------------------------------------
    # Experiment 1: Original Images
    # -----------------------------------------------------

    baseline_best = train_experiment(
        experiment_name="baseline",
        data_path=BASELINE_DATA
    )

    baseline_results = evaluate_experiment(
        experiment_name="baseline",
        best_model_path=baseline_best,
        data_path=BASELINE_DATA
    )

    all_results.append(
        baseline_results
    )

    # -----------------------------------------------------
    # Experiment 2: CLAHE Images
    # -----------------------------------------------------

    clahe_best = train_experiment(
        experiment_name="clahe",
        data_path=CLAHE_DATA
    )

    clahe_results = evaluate_experiment(
        experiment_name="clahe",
        best_model_path=clahe_best,
        data_path=CLAHE_DATA
    )

    all_results.append(
        clahe_results
    )

    # -----------------------------------------------------
    # Save comparison
    # -----------------------------------------------------

    save_results(
        all_results
    )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)