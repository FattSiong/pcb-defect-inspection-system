from ultralytics import YOLO
from pathlib import Path
import torch


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced\data.yaml"
)

# Maximum number of epochs for the experiment
EPOCHS = 100

IMAGE_SIZE = 640

# Automatically determine a suitable batch size
# based on available GPU memory
BATCH_SIZE = -1

# GPU 0
DEVICE = 0

# Keep random behaviour consistent
SEED = 42


# =========================================================
# OUTPUT LOCATION
# =========================================================

PROJECT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system\runs\epoch_testing"
)

EXPERIMENT_NAME = "reduced_dataset_100_epochs"


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Please check the PyTorch GPU installation."
        )

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )


# =========================================================
# CHECK DATASET
# =========================================================

def check_dataset():

    print("\n" + "=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    print(
        f"Dataset:\n{DATASET}"
    )

    if not DATASET.exists():

        raise FileNotFoundError(
            f"\ndata.yaml not found:\n"
            f"{DATASET}"
        )

    print("\nDataset found successfully.")


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    print("\n" + "=" * 70)
    print("EPOCH DETERMINATION EXPERIMENT")
    print("=" * 70)

    print(f"Model      : {MODEL_NAME}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Device     : {DEVICE}")
    print(f"Seed       : {SEED}")

    print("\nTraining will run for all 50 epochs.")
    print(
        "Validation performance will be recorded "
        "after every epoch."
    )

    print("=" * 70)


    # Start from pretrained YOLOv8n
    model = YOLO(
        MODEL_NAME
    )


    # =====================================================
    # TRAIN
    # =====================================================

    model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        seed=SEED,

        patience=150,

        project=str(PROJECT_DIR),

        name=EXPERIMENT_NAME,

        exist_ok=True,

        plots=True
    )


    # =====================================================
    # GET ACTUAL YOLO SAVE LOCATION
    # =====================================================

    save_dir = Path(
        model.trainer.save_dir
    )

    best_model = Path(
        model.trainer.best
    )

    last_model = Path(
        model.trainer.last
    )

    results_csv = (
        save_dir /
        "results.csv"
    )


    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nResults directory:\n"
        f"{save_dir}"
    )

    print(
        f"\nBest model:\n"
        f"{best_model}"
    )

    print(
        f"\nLast model:\n"
        f"{last_model}"
    )

    print(
        f"\nEpoch results:\n"
        f"{results_csv}"
    )

# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    check_dataset()

    train_model()


if __name__ == "__main__":
    main()