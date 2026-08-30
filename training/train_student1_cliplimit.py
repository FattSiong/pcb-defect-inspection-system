from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# YOLO SETTINGS
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

BLOCK_SIZE = 41
ADAPTIVE_C = 2

CLAHE_TILE_GRID_SIZE = "8x8"

CLIP_LIMITS = [
    1.0,
    2.0,
    3.0
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

NEW_DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_cliplimit"
)

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student1_cliplimit"
)

COMPARISON_CSV = (
    RUNS_ROOT /
    "cliplimit_comparison.csv"
)


# =========================================================
# EXISTING clipLimit = 2.0
# =========================================================

# Exact configuration:
# clipLimit = 2.0
# tileGrid = 8x8
# blockSize = 41
# C = 2

EXISTING_CLIP2_DATASET = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_blocksize"
    r"\block_41"
    r"\data.yaml"
)

EXISTING_CLIP2_MODEL = (
    PROJECT_ROOT /
    "runs" /
    "student1_blocksize" /
    "block_41" /
    "weights" /
    "best.pt"
)


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 90)
    print("STUDENT 1 - CLAHE CLIPLIMIT TUNING")
    print("=" * 90)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        f"\nblockSize : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C         : "
        f"{ADAPTIVE_C}"
    )

    print(
        f"tileGrid  : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"clipLimits: "
        f"{CLIP_LIMITS}"
    )

    print(
        f"Epochs    : "
        f"{EPOCHS}"
    )


# =========================================================
# GET DATASET
# =========================================================

def get_dataset_yaml(
    clip_limit
):

    # Existing 2.0 configuration
    if clip_limit == 2.0:

        return (
            EXISTING_CLIP2_DATASET
        )


    return (
        NEW_DATASET_ROOT /
        f"clip_{clip_limit:.1f}" /
        "data.yaml"
    )


# =========================================================
# CHECK FILES
# =========================================================

def check_required_files():

    print("\n" + "=" * 90)
    print("CHECKING FILES")
    print("=" * 90)


    if not EXISTING_CLIP2_DATASET.exists():

        raise FileNotFoundError(
            f"Existing clipLimit=2.0 "
            f"dataset not found:\n"
            f"{EXISTING_CLIP2_DATASET}"
        )


    if not EXISTING_CLIP2_MODEL.exists():

        raise FileNotFoundError(
            f"Existing clipLimit=2.0 "
            f"best.pt not found:\n"
            f"{EXISTING_CLIP2_MODEL}"
        )


    print(
        "\nclipLimit=2.0 model:"
    )

    print(
        EXISTING_CLIP2_MODEL
    )


    for clip_limit in [
        1.0,
        3.0
    ]:

        yaml_path = get_dataset_yaml(
            clip_limit
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\nclipLimit="
                f"{clip_limit} "
                f"dataset missing:\n"
                f"{yaml_path}\n\n"
                f"Run preprocessing first."
            )

        print(
            f"\nclipLimit="
            f"{clip_limit}:"
        )

        print(
            yaml_path
        )


# =========================================================
# TRAIN NEW MODEL
# =========================================================

def train_configuration(
    clip_limit
):

    yaml_path = get_dataset_yaml(
        clip_limit
    )

    experiment_name = (
        f"clip_{clip_limit:.1f}"
    )


    print("\n" + "=" * 90)

    print(
        f"TRAINING clipLimit = "
        f"{clip_limit}"
    )

    print("=" * 90)


    # Fresh YOLOv8n for fair comparison
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


    if not best_model.exists():

        raise FileNotFoundError(
            f"Training completed but best.pt "
            f"was not found:\n"
            f"{best_model}"
        )


    print(
        f"\nTraining time: "
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
# VALIDATE
# =========================================================

def validate_model(
    clip_limit,
    yaml_path,
    model_path,
    training_time=0.0
):

    print("\n" + "-" * 90)

    print(
        f"VALIDATING clipLimit = "
        f"{clip_limit}"
    )

    print("-" * 90)


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
            f"clip_{clip_limit:.1f}"
            f"_validation"
        ),

        exist_ok=True,

        plots=True
    )


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
        precision
        + recall
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
        f"\nPrecision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.4f}"
    )

    print(
        f"mAP50     : "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95  : "
        f"{map50_95:.4f}"
    )


    return {
        "Clip Limit":
            clip_limit,

        "Block Size":
            BLOCK_SIZE,

        "C":
            ADAPTIVE_C,

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
            training_time,

        "Best Model":
            str(model_path)
    }


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    results
):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = sorted(
        results,
        key=lambda x:
            x["Clip Limit"]
    )


    fieldnames = [
        "Clip Limit",
        "Block Size",
        "C",
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
        f"\nResults saved:\n"
        f"{COMPARISON_CSV}"
    )


# =========================================================
# PRINT COMPARISON
# =========================================================

def print_comparison(
    results
):

    results = sorted(
        results,
        key=lambda x:
            x["Clip Limit"]
    )


    print("\n" + "=" * 105)
    print("CLAHE CLIPLIMIT VALIDATION COMPARISON")
    print("=" * 105)


    print(
        f"{'Clip Limit':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'mAP50':<14}"
        f"{'mAP50-95':<14}"
    )

    print(
        "-" * 105
    )


    for result in results:

        print(
            f"{result['Clip Limit']:<14.1f}"
            f"{result['Precision']:<14.4f}"
            f"{result['Recall']:<14.4f}"
            f"{result['F1-score']:<14.4f}"
            f"{result['mAP50']:<14.4f}"
            f"{result['mAP50-95']:<14.4f}"
        )


    best = max(
        results,
        key=lambda x:
            x["mAP50-95"]
    )


    print("\n" + "=" * 105)
    print("BEST CLIP LIMIT")
    print("=" * 105)

    print(
        f"clipLimit : "
        f"{best['Clip Limit']}"
    )

    print(
        f"blockSize : "
        f"{best['Block Size']}"
    )

    print(
        f"C         : "
        f"{best['C']}"
    )

    print(
        f"Precision : "
        f"{best['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best['Recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{best['F1-score']:.4f}"
    )

    print(
        f"mAP50     : "
        f"{best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95  : "
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
    # clipLimit = 2.0
    # REUSE EXISTING MODEL
    # =====================================================

    print("\n" + "=" * 90)
    print("REUSING EXISTING clipLimit = 2.0 MODEL")
    print("=" * 90)

    existing_result = validate_model(

        clip_limit=2.0,

        yaml_path=EXISTING_CLIP2_DATASET,

        model_path=EXISTING_CLIP2_MODEL,

        training_time=0.0
    )

    results.append(
        existing_result
    )

    save_results(
        results
    )


    # =====================================================
    # NEW clipLimit 1.0 AND 3.0
    # =====================================================

    for clip_limit in [
        1.0,
        3.0
    ]:

        (
            yaml_path,
            best_model,
            training_time
        ) = train_configuration(
            clip_limit
        )


        result = validate_model(
            clip_limit,
            yaml_path,
            best_model,
            training_time
        )


        results.append(
            result
        )


        # Save immediately after each model
        save_results(
            results
        )


    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print_comparison(
        results
    )


    print("\n" + "=" * 105)
    print("CLIPLIMIT PARAMETER TUNING COMPLETED")
    print("=" * 105)


if __name__ == "__main__":
    main()