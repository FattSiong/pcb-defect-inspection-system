from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# GENERAL SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640

TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4

DEVICE = 0
SEED = 42


# =========================================================
# FIXED STUDENT 1 PARAMETERS
# =========================================================

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = "8x8"

BLOCK_SIZE = 41

C_VALUES = [
    2,
    5,
    8
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)


# ---------------------------------------------------------
# Existing C = 2 dataset
#
# C=2 + blockSize=41 was already produced during
# the block-size experiment.
# ---------------------------------------------------------

C2_DATASET_YAML = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_blocksize"
    r"\block_41"
    r"\data.yaml"
)


# Existing trained C=2 / blockSize=41 model
C2_MODEL_PATH = (
    PROJECT_ROOT /
    "runs" /
    "student1_blocksize" /
    "block_41" /
    "weights" /
    "best.pt"
)


# ---------------------------------------------------------
# New C = 5 and C = 8 datasets
# ---------------------------------------------------------

C_DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_c"
)


# ---------------------------------------------------------
# Experiment output
# ---------------------------------------------------------

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student1_c"
)

COMPARISON_CSV = (
    RUNS_ROOT /
    "c_comparison.csv"
)


# =========================================================
# ENVIRONMENT CHECK
# =========================================================

def check_environment():

    print("=" * 90)
    print("STUDENT 1 - ADAPTIVE GAUSSIAN C PARAMETER TUNING")
    print("=" * 90)


    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "CUDA available  :",
        torch.cuda.is_available()
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )


    print(
        f"\nFixed blockSize : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"CLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tile grid : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"Epochs          : "
        f"{EPOCHS}"
    )

    print(
        f"C values        : "
        f"{C_VALUES}"
    )


# =========================================================
# GET DATASET YAML
# =========================================================

def get_dataset_yaml(c_value):

    # -----------------------------------------------------
    # C = 2 uses existing block-size dataset
    # -----------------------------------------------------

    if c_value == 2:

        return C2_DATASET_YAML


    # -----------------------------------------------------
    # C = 5 / 8 use newly processed datasets
    # -----------------------------------------------------

    return (
        C_DATASET_ROOT /
        f"c_{c_value}" /
        "data.yaml"
    )


# =========================================================
# VERIFY FILES
# =========================================================

def check_required_files():

    print("\n" + "=" * 90)
    print("FILE CHECK")
    print("=" * 90)


    if not C2_DATASET_YAML.exists():

        raise FileNotFoundError(
            f"C=2 dataset not found:\n"
            f"{C2_DATASET_YAML}"
        )


    if not C2_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"C=2 existing best.pt not found:\n"
            f"{C2_MODEL_PATH}"
        )


    print(
        "\nC=2 existing model found:"
    )

    print(
        C2_MODEL_PATH
    )


    for c_value in [
        5,
        8
    ]:

        yaml_path = get_dataset_yaml(
            c_value
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\nC={c_value} dataset not found:\n"
                f"{yaml_path}\n\n"
                f"Run preprocess_student1_c.py first."
            )


        print(
            f"\nC={c_value} dataset found:"
        )

        print(
            yaml_path
        )


# =========================================================
# TRAIN NEW CONFIGURATION
# =========================================================

def train_new_configuration(c_value):

    yaml_path = get_dataset_yaml(
        c_value
    )


    experiment_name = (
        f"c_{c_value}"
    )


    print("\n" + "=" * 90)

    print(
        f"TRAINING C = {c_value}"
    )

    print("=" * 90)


    print(
        f"blockSize : {BLOCK_SIZE}"
    )

    print(
        f"C         : {c_value}"
    )

    print(
        f"Epochs    : {EPOCHS}"
    )


    # -----------------------------------------------------
    # IMPORTANT:
    # Fresh pretrained YOLOv8n for each parameter test.
    # -----------------------------------------------------

    model = YOLO(
        MODEL_NAME
    )


    start_time = time.time()


    model.train(

        data=str(
            yaml_path
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=TRAIN_BATCH_SIZE,

        device=DEVICE,

        seed=SEED,

        # Ensure all 100 epochs can run
        patience=150,

        project=str(
            RUNS_ROOT
        ),

        name=experiment_name,

        exist_ok=True,

        plots=True
    )


    training_time = (
        time.time()
        - start_time
    )


    best_model = Path(
        model.trainer.best
    )


    print(
        "\nTraining completed."
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"\nBest model:\n"
        f"{best_model}"
    )


    return (
        yaml_path,
        best_model,
        training_time
    )


# =========================================================
# VALIDATE ONE MODEL
# =========================================================

def validate_model(
    c_value,
    yaml_path,
    model_path,
    training_time=None
):

    print("\n" + "-" * 90)

    print(
        f"VALIDATING C = {c_value}"
    )

    print("-" * 90)


    print(
        f"Model:\n{model_path}"
    )


    model = YOLO(
        str(model_path)
    )


    metrics = model.val(

        data=str(
            yaml_path
        ),

        split="val",

        imgsz=IMAGE_SIZE,

        batch=VAL_BATCH_SIZE,

        device=DEVICE,

        project=str(
            RUNS_ROOT
        ),

        name=(
            f"c_{c_value}_validation"
        ),

        exist_ok=True,

        plots=True
    )


    # =====================================================
    # EXTRACT METRICS
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


    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    print(
        f"\nPrecision : {precision:.4f}"
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


    return {
        "C":
            c_value,

        "Block Size":
            BLOCK_SIZE,

        "CLAHE Clip Limit":
            CLAHE_CLIP_LIMIT,

        "Tile Grid":
            CLAHE_TILE_GRID_SIZE,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1-score":
            f1,

        "mAP50":
            map50,

        "mAP50-95":
            map50_95,

        "Training Time (seconds)":
            (
                training_time
                if training_time is not None
                else 0.0
            ),

        "Best Model":
            str(model_path)
    }


# =========================================================
# SAVE COMPARISON CSV
# =========================================================

def save_results(results):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [
        "C",
        "Block Size",
        "CLAHE Clip Limit",
        "Tile Grid",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
        "Training Time (seconds)",
        "Best Model"
    ]


    with open(
        COMPARISON_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        f"\nComparison saved to:\n"
        f"{COMPARISON_CSV}"
    )


# =========================================================
# PRINT FINAL COMPARISON
# =========================================================

def print_comparison(results):

    results = sorted(
        results,
        key=lambda item:
            item["C"]
    )


    print("\n" + "=" * 100)
    print("C PARAMETER VALIDATION COMPARISON")
    print("=" * 100)


    print(
        f"{'C':<8}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'mAP50':<14}"
        f"{'mAP50-95':<14}"
    )


    print(
        "-" * 100
    )


    for result in results:

        print(
            f"{result['C']:<8}"
            f"{result['Precision']:<14.4f}"
            f"{result['Recall']:<14.4f}"
            f"{result['F1-score']:<14.4f}"
            f"{result['mAP50']:<14.4f}"
            f"{result['mAP50-95']:<14.4f}"
        )


    # =====================================================
    # FIND BEST C USING mAP50-95
    # =====================================================

    best = max(
        results,
        key=lambda result:
            result["mAP50-95"]
    )


    print("\n" + "=" * 100)
    print("BEST C VALUE")
    print("=" * 100)


    print(
        f"C          : "
        f"{best['C']}"
    )

    print(
        f"Block Size : "
        f"{best['Block Size']}"
    )

    print(
        f"Precision  : "
        f"{best['Precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best['Recall']:.4f}"
    )

    print(
        f"F1-score   : "
        f"{best['F1-score']:.4f}"
    )

    print(
        f"mAP50      : "
        f"{best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95   : "
        f"{best['mAP50-95']:.4f}"
    )


    print(
        "\nBest model:"
    )

    print(
        best["Best Model"]
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    check_required_files()


    results = []


    # =====================================================
    # C = 2
    #
    # DO NOT RETRAIN.
    # Reuse existing blockSize 41 model.
    # =====================================================

    print("\n" + "=" * 90)
    print("REUSING EXISTING C = 2 MODEL")
    print("=" * 90)


    c2_result = validate_model(
        c_value=2,
        yaml_path=C2_DATASET_YAML,
        model_path=C2_MODEL_PATH,
        training_time=None
    )


    results.append(
        c2_result
    )


    # Save immediately
    save_results(
        results
    )


    # =====================================================
    # C = 5 AND C = 8
    # =====================================================

    for c_value in [
        5,
        8
    ]:

        (
            yaml_path,
            best_model,
            training_time
        ) = train_new_configuration(
            c_value
        )


        result = validate_model(
            c_value,
            yaml_path,
            best_model,
            training_time
        )


        results.append(
            result
        )


        # Save after every completed configuration
        save_results(
            results
        )


    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print_comparison(
        results
    )


    print("\n" + "=" * 100)
    print("C PARAMETER TUNING COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()