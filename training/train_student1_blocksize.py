from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

PROCESSED_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_student1_blocksize"
)

RUNS_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_blocksize"
)


# =========================================================
# PARAMETERS BEING COMPARED
# =========================================================

BLOCK_SIZES = [
    41, 51
]


# =========================================================
# FIXED YOLO SETTINGS
# =========================================================

EPOCHS = 100
IMAGE_SIZE = 640

TRAIN_BATCH_SIZE = -1

# model.val() needs positive integer
VAL_BATCH_SIZE = 4

DEVICE = 0
SEED = 42


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 80)
    print("STUDENT 1 BLOCK SIZE TRAINING")
    print("=" * 80)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Block sizes:",
        BLOCK_SIZES
    )


# =========================================================
# TRAIN ONE CONFIGURATION
# =========================================================

def train_block_size(
    block_size
):

    dataset = (
        PROCESSED_ROOT /
        f"block_{block_size}"
    )

    yaml_path = (
        dataset /
        "data.yaml"
    )


    if not yaml_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{yaml_path}\n\n"
            f"Run preprocessing first."
        )


    experiment_name = (
        f"block_{block_size}"
    )


    print("\n" + "=" * 80)
    print(
        f"TRAINING BLOCK SIZE {block_size}"
    )
    print("=" * 80)


    # -----------------------------------------------------
    # Fresh pretrained model for every block size
    # -----------------------------------------------------

    model = YOLO(
        MODEL_NAME
    )


    start_time = time.time()


    model.train(

        data=str(yaml_path),

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


    print(
        "\nBest model:"
    )

    print(
        best_model
    )


    return (
        yaml_path,
        best_model,
        training_time
    )


# =========================================================
# VALIDATE BEST MODEL
# =========================================================

def validate_model(
    block_size,
    yaml_path,
    best_model,
    training_time
):

    print("\n" + "-" * 80)
    print(
        f"VALIDATING BLOCK SIZE {block_size}"
    )
    print("-" * 80)


    model = YOLO(
        str(best_model)
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
            f"block_{block_size}_validation"
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


    return {
        "Block Size":
            block_size,

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
            str(best_model)
    }


# =========================================================
# SAVE COMPARISON
# =========================================================

def save_results(results):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (
        RUNS_ROOT /
        "blocksize_comparison.csv"
    )


    fieldnames = [
        "Block Size",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
        "Training Time (seconds)",
        "Best Model"
    ]


    with open(
        output_file,
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
        f"{output_file}"
    )


# =========================================================
# PRINT FINAL COMPARISON
# =========================================================

def print_comparison(results):

    print("\n" + "=" * 95)
    print("BLOCK SIZE VALIDATION COMPARISON")
    print("=" * 95)

    print(
        f"{'Block':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<12}"
    )

    print(
        "-" * 95
    )


    for result in results:

        print(
            f"{result['Block Size']:<10}"
            f"{result['Precision']:<12.4f}"
            f"{result['Recall']:<12.4f}"
            f"{result['F1-score']:<12.4f}"
            f"{result['mAP50']:<12.4f}"
            f"{result['mAP50-95']:<12.4f}"
        )


    # Primary selection metric
    best = max(
        results,
        key=lambda result:
            result["mAP50-95"]
    )


    print("\n" + "=" * 95)
    print("BEST BLOCK SIZE")
    print("=" * 95)

    print(
        f"Block Size : "
        f"{best['Block Size']}"
    )

    print(
        f"mAP50-95   : "
        f"{best['mAP50-95']:.4f}"
    )

    print(
        f"mAP50      : "
        f"{best['mAP50']:.4f}"
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

    all_results = []


    for block_size in BLOCK_SIZES:

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        (
            yaml_path,
            best_model,
            training_time
        ) = train_block_size(
            block_size
        )


        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        result = validate_model(
            block_size,
            yaml_path,
            best_model,
            training_time
        )


        all_results.append(
            result
        )


        # Save after every run
        save_results(
            all_results
        )


    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print_comparison(
        all_results
    )


if __name__ == "__main__":
    main()