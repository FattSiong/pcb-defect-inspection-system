from ultralytics import YOLO
from pathlib import Path
import csv
import torch


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 1
IMAGE_SIZE = 640

# Recommended for your RTX 3050 4GB.
# Ultralytics will automatically choose a suitable batch size.
BATCH_SIZE = -1

DEVICE = 0
SEED = 42
PATIENCE = 10


# =========================================================
# CLAHE DATASETS
# =========================================================

CLAHE_DATASETS = {
    1.0: Path(
        r"C:\Users\tiong\Downloads\pcb_dataset_clahe_1.0\data.yaml"
    ),

    2.0: Path(
        r"C:\Users\tiong\Downloads\pcb_dataset_clahe_2.0\data.yaml"
    ),

    3.0: Path(
        r"C:\Users\tiong\Downloads\pcb_dataset_clahe_3.0\data.yaml"
    ),
}


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

PROJECT_DIR = Path(
    "runs/clahe_parameter_tuning"
)


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Training would run on CPU."
        )

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()


# =========================================================
# CHECK DATASETS
# =========================================================

def check_datasets():

    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    for clip_limit, yaml_path in CLAHE_DATASETS.items():

        print(
            f"CLAHE {clip_limit}: "
            f"{yaml_path}"
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\ndata.yaml not found:\n"
                f"{yaml_path}"
            )

    print("\nAll dataset YAML files found.\n")


# =========================================================
# CALCULATE F1
# =========================================================

def calculate_f1(precision, recall):

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# =========================================================
# TRAIN ONE CLAHE CONFIGURATION
# =========================================================

def train_clahe(
    clip_limit,
    data_yaml
):

    experiment_name = (
        f"clahe_clip_{clip_limit}"
    )

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING CLAHE clipLimit = {clip_limit}"
    )
    print("=" * 70)

    print(f"Dataset     : {data_yaml}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Batch       : {BATCH_SIZE}")
    print(f"Device      : {DEVICE}")
    print(f"Seed        : {SEED}")
    print(f"Patience    : {PATIENCE}")

    print("=" * 70)


    # IMPORTANT:
    # Start from the SAME pretrained YOLOv8n weights
    # for every CLAHE experiment.
    model = YOLO(
        MODEL_NAME
    )


    model.train(

        data=str(data_yaml),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        seed=SEED,

        patience=PATIENCE,

        project=str(PROJECT_DIR),

        name=experiment_name,

        exist_ok=True
    )


    # =====================================================
    # FIND BEST.PT
    # =====================================================

    best_model_path = Path(
    model.trainer.best
    )

    if not best_model_path.exists():

        raise FileNotFoundError(
            f"\nbest.pt not found:\n"
            f"{best_model_path}"
        )


    print("\nTraining completed.")

    print(
        f"Best model:\n"
        f"{best_model_path}"
    )

    return best_model_path


# =========================================================
# VALIDATE BEST MODEL
# =========================================================

def validate_model(
    clip_limit,
    data_yaml,
    best_model_path
):

    print("\n")
    print("=" * 70)
    print(
        f"VALIDATING CLAHE clipLimit = {clip_limit}"
    )
    print("=" * 70)


    # Load best checkpoint
    model = YOLO(
        str(best_model_path)
    )


    # Explicitly evaluate on VAL dataset
    metrics = model.val(

        data=str(data_yaml),

        split="val",

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        project=str(PROJECT_DIR),

        name=f"val_clip_{clip_limit}"
    )


    # =====================================================
    # GET METRICS
    # =====================================================

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map50_95 = float(
        metrics.box.map
    )

    f1 = calculate_f1(
        precision,
        recall
    )


    print("\n")
    print("-" * 70)

    print(
        f"CLAHE clipLimit = {clip_limit}"
    )

    print("-" * 70)

    print(
        f"Precision    : "
        f"{precision:.6f}"
    )

    print(
        f"Recall       : "
        f"{recall:.6f}"
    )

    print(
        f"F1-score     : "
        f"{f1:.6f}"
    )

    print(
        f"mAP50        : "
        f"{map50:.6f}"
    )

    print(
        f"mAP50-95     : "
        f"{map50_95:.6f}"
    )

    print("-" * 70)


    return {
        "clip_limit": clip_limit,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "best_model": str(
            best_model_path
        )
    }


# =========================================================
# SAVE RESULTS TO CSV
# =========================================================

def save_results(results):

    PROJECT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROJECT_DIR
        / "clahe_parameter_results.csv"
    )


    fieldnames = [
        "clip_limit",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "best_model"
    ]


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


# =========================================================
# PRINT COMPARISON
# =========================================================

def print_comparison(results):

    print("\n")
    print("=" * 90)
    print("CLAHE PARAMETER COMPARISON")
    print("=" * 90)

    print(
        f"{'ClipLimit':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'mAP50':<14}"
        f"{'mAP50-95':<14}"
    )

    print("-" * 90)


    for result in results:

        print(
            f"{result['clip_limit']:<12.1f}"
            f"{result['precision']:<14.4f}"
            f"{result['recall']:<14.4f}"
            f"{result['f1']:<14.4f}"
            f"{result['map50']:<14.4f}"
            f"{result['map50_95']:<14.4f}"
        )


    # =====================================================
    # SELECT BEST PARAMETER
    # =====================================================

    best_result = max(
        results,
        key=lambda x: x["map50_95"]
    )


    print("\n")
    print("=" * 90)
    print("BEST CLAHE PARAMETER")
    print("=" * 90)

    print(
        f"Clip Limit : "
        f"{best_result['clip_limit']}"
    )

    print(
        f"Precision  : "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1-score   : "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"mAP50      : "
        f"{best_result['map50']:.4f}"
    )

    print(
        f"mAP50-95   : "
        f"{best_result['map50_95']:.4f}"
    )

    print(
        f"\nBest model:\n"
        f"{best_result['best_model']}"
    )

    print("=" * 90)


    return best_result


# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    check_datasets()


    all_results = []


    # =====================================================
    # RUN ALL THREE CLAHE PARAMETERS
    # =====================================================

    for clip_limit, data_yaml in CLAHE_DATASETS.items():

        # Train
        best_model_path = train_clahe(
            clip_limit,
            data_yaml
        )


        # Validate best.pt
        result = validate_model(
            clip_limit,
            data_yaml,
            best_model_path
        )


        all_results.append(
            result
        )


    # =====================================================
    # SAVE AND COMPARE
    # =====================================================

    save_results(
        all_results
    )


    best_result = print_comparison(
        all_results
    )


    print("\nParameter tuning completed.")

    print(
        "\nDo NOT use the test dataset "
        "to choose the CLAHE parameter."
    )

    print(
        "The selected parameter above "
        "was chosen using validation performance."
    )

    print(
        "\nThe next stage is to evaluate ONLY "
        "this selected best.pt on the test dataset."
    )


if __name__ == "__main__":

    main()