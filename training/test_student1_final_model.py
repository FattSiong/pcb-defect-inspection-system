from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import json
import numpy as np


# =========================================================
# FINAL CONFIGURATION
# =========================================================

CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = "8x8"

BLOCK_SIZE = 41
ADAPTIVE_C = 2

IMAGE_SIZE = 640
TEST_BATCH_SIZE = 4
DEVICE = 0


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)


# Final TEST dataset
FINAL_DATASET = Path(
    r"C:\Users\tiong\Downloads"
    r"\pcb_dataset_student1_final_test"
)

FINAL_DATA_YAML = (
    FINAL_DATASET /
    "data.yaml"
)


# =========================================================
# WINNING MODEL
# =========================================================

# clipLimit = 1.0
# blockSize = 41
# C = 2
#
# This is the model already trained during
# clipLimit optimisation.

FINAL_MODEL = (
    PROJECT_ROOT /
    "runs" /
    "student1_cliplimit" /
    "clip_1.0" /
    "weights" /
    "best.pt"
)


# =========================================================
# OUTPUT
# =========================================================

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student1_final_test"
)

EVALUATION_NAME = (
    "technique_a_test"
)

EVALUATION_DIR = (
    RUNS_ROOT /
    EVALUATION_NAME
)


PREPROCESSING_SUMMARY = (
    RUNS_ROOT /
    "final_test_preprocessing_summary.csv"
)


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 100)
    print("STUDENT 1 - FINAL TECHNIQUE A TEST")
    print("=" * 100)


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
            "CUDA is not available."
        )


    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )


    print("\nFINAL LOCKED CONFIGURATION")

    print(
        f"CLAHE clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"CLAHE tileGrid  : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"blockSize       : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C               : "
        f"{ADAPTIVE_C}"
    )

    print(
        f"Image size      : "
        f"{IMAGE_SIZE}"
    )


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

def check_files():

    print("\n" + "=" * 100)
    print("CHECKING FINAL TEST FILES")
    print("=" * 100)


    if not FINAL_MODEL.exists():

        raise FileNotFoundError(
            f"Final best.pt not found:\n"
            f"{FINAL_MODEL}"
        )


    if not FINAL_DATA_YAML.exists():

        raise FileNotFoundError(
            f"Final data.yaml not found:\n"
            f"{FINAL_DATA_YAML}\n\n"
            f"Run preprocess_student1_final_test.py first."
        )


    print(
        f"\nFinal model:\n"
        f"{FINAL_MODEL}"
    )

    print(
        f"\nFinal dataset:\n"
        f"{FINAL_DATA_YAML}"
    )


# =========================================================
# COUNT TEST IMAGES
# =========================================================

def count_test_images():

    image_folder = (
        FINAL_DATASET /
        "test" /
        "images"
    )


    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }


    return len([
        path
        for path in image_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        )
    ])


# =========================================================
# LOAD CLASSICAL PREPROCESSING SPEED
# =========================================================

def load_preprocessing_speed():

    if not PREPROCESSING_SUMMARY.exists():

        print(
            "\nWARNING: Preprocessing timing "
            "summary was not found."
        )

        return 0.0


    with open(
        PREPROCESSING_SUMMARY,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        row = next(
            reader,
            None
        )


    if row is None:

        return 0.0


    try:

        return float(
            row[
                "Average Preprocessing Time (ms/image)"
            ]
        )

    except (
        KeyError,
        ValueError
    ):

        return 0.0


# =========================================================
# RUN FINAL TEST
# =========================================================

def run_test():

    print("\n" + "=" * 100)
    print("RUNNING FINAL TEST EVALUATION")
    print("=" * 100)


    model = YOLO(
        str(FINAL_MODEL)
    )


    # =====================================================
    # IMPORTANT:
    #
    # split="test"
    #
    # NO TRAINING IS PERFORMED HERE.
    # =====================================================

    metrics = model.val(

        data=str(
            FINAL_DATA_YAML
        ),

        split="test",

        imgsz=IMAGE_SIZE,

        batch=TEST_BATCH_SIZE,

        device=DEVICE,

        project=str(
            RUNS_ROOT
        ),

        name=EVALUATION_NAME,

        exist_ok=True,

        plots=True
    )


    return (
        model,
        metrics
    )


# =========================================================
# EXTRACT OVERALL METRICS
# =========================================================

def extract_overall_metrics(
    metrics,
    test_image_count
):

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


    return {
        "Technique":
            "CLAHE + Adaptive Gaussian Thresholding",

        "CLAHE Clip Limit":
            CLAHE_CLIP_LIMIT,

        "CLAHE Tile Grid":
            CLAHE_TILE_GRID_SIZE,

        "Adaptive Block Size":
            BLOCK_SIZE,

        "Adaptive C":
            ADAPTIVE_C,

        "Test Images":
            test_image_count,

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

        "Model":
            str(FINAL_MODEL),

        "Dataset":
            str(FINAL_DATA_YAML)
    }


# =========================================================
# SAVE OVERALL SUMMARY
# =========================================================

def save_overall_summary(
    summary
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_path = (
        EVALUATION_DIR /
        "test_metrics_summary.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary.keys()
            )
        )

        writer.writeheader()

        writer.writerow(
            summary
        )


    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    json_path = (
        EVALUATION_DIR /
        "test_metrics_summary.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    print(
        f"\nOverall CSV:\n"
        f"{csv_path}"
    )

    print(
        f"\nOverall JSON:\n"
        f"{json_path}"
    )


# =========================================================
# SAVE PER-CLASS METRICS
# =========================================================

def save_per_class_metrics(
    model,
    metrics
):

    box = metrics.box


    # -----------------------------------------------------
    # Class IDs that were evaluated
    # -----------------------------------------------------

    if hasattr(
        box,
        "ap_class_index"
    ):

        class_ids = np.asarray(
            box.ap_class_index
        ).astype(int)

    else:

        class_ids = np.arange(
            len(
                np.asarray(
                    box.ap50
                )
            )
        )


    # -----------------------------------------------------
    # Per-class arrays
    # -----------------------------------------------------

    precision_values = np.asarray(
        box.p
    )

    recall_values = np.asarray(
        box.r
    )

    f1_values = np.asarray(
        box.f1
    )

    map50_values = np.asarray(
        box.ap50
    )

    map50_95_values = np.asarray(
        box.ap
    )


    rows = []


    for index, class_id in enumerate(
        class_ids
    ):

        class_name = model.names[
            int(class_id)
        ]


        rows.append({
            "Class ID":
                int(class_id),

            "Class Name":
                class_name,

            "Precision":
                float(
                    precision_values[index]
                ),

            "Recall":
                float(
                    recall_values[index]
                ),

            "F1-score":
                float(
                    f1_values[index]
                ),

            "mAP50":
                float(
                    map50_values[index]
                ),

            "mAP50-95":
                float(
                    map50_95_values[index]
                )
        })


    output_path = (
        EVALUATION_DIR /
        "test_metrics_by_class.csv"
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Class ID",
            "Class Name",
            "Precision",
            "Recall",
            "F1-score",
            "mAP50",
            "mAP50-95"
        ]


        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print(
        f"\nPer-class metrics:\n"
        f"{output_path}"
    )


# =========================================================
# SAVE SPEED METRICS
# =========================================================

def save_speed_metrics(
    metrics
):

    # Ultralytics speed values are milliseconds per image
    speed = metrics.speed


    yolo_preprocess_ms = float(
        speed.get(
            "preprocess",
            0.0
        )
    )

    inference_ms = float(
        speed.get(
            "inference",
            0.0
        )
    )

    postprocess_ms = float(
        speed.get(
            "postprocess",
            0.0
        )
    )


    # Classical image-processing pipeline
    classical_preprocessing_ms = (
        load_preprocessing_speed()
    )


    # YOLO pipeline total
    yolo_total_ms = (
        yolo_preprocess_ms
        + inference_ms
        + postprocess_ms
    )


    # Complete Technique A pipeline:
    #
    # CLAHE + Adaptive Thresholding
    # +
    # YOLO preprocessing/inference/postprocessing
    end_to_end_ms = (
        classical_preprocessing_ms
        + yolo_total_ms
    )


    if end_to_end_ms > 0:

        approximate_fps = (
            1000.0 /
            end_to_end_ms
        )

    else:

        approximate_fps = 0.0


    rows = [
        {
            "Stage":
                "CLAHE + Adaptive Gaussian preprocessing",

            "Milliseconds per Image":
                classical_preprocessing_ms
        },

        {
            "Stage":
                "YOLO internal preprocessing",

            "Milliseconds per Image":
                yolo_preprocess_ms
        },

        {
            "Stage":
                "YOLO inference",

            "Milliseconds per Image":
                inference_ms
        },

        {
            "Stage":
                "YOLO postprocessing",

            "Milliseconds per Image":
                postprocess_ms
        },

        {
            "Stage":
                "Complete Technique A pipeline",

            "Milliseconds per Image":
                end_to_end_ms
        },

        {
            "Stage":
                "Approximate Complete Pipeline FPS",

            "Milliseconds per Image":
                approximate_fps
        }
    ]


    output_path = (
        EVALUATION_DIR /
        "test_processing_speed.csv"
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Stage",
                "Milliseconds per Image"
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print(
        f"\nProcessing speed:\n"
        f"{output_path}"
    )


    print("\nSPEED SUMMARY")

    print(
        f"Classical preprocessing : "
        f"{classical_preprocessing_ms:.3f} ms/image"
    )

    print(
        f"YOLO inference          : "
        f"{inference_ms:.3f} ms/image"
    )

    print(
        f"Complete pipeline       : "
        f"{end_to_end_ms:.3f} ms/image"
    )

    print(
        f"Approximate FPS         : "
        f"{approximate_fps:.2f}"
    )


# =========================================================
# PRINT FINAL RESULT
# =========================================================

def print_final_result(
    summary
):

    print("\n" + "=" * 100)
    print("FINAL STUDENT 1 TECHNIQUE A TEST RESULT")
    print("=" * 100)


    print(
        f"Precision : "
        f"{summary['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{summary['Recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{summary['F1-score']:.4f}"
    )

    print(
        f"mAP50     : "
        f"{summary['mAP50']:.4f}"
    )

    print(
        f"mAP50-95  : "
        f"{summary['mAP50-95']:.4f}"
    )


    print("\nFINAL PARAMETERS")

    print(
        f"clipLimit : "
        f"{CLAHE_CLIP_LIMIT}"
    )

    print(
        f"tileGrid  : "
        f"{CLAHE_TILE_GRID_SIZE}"
    )

    print(
        f"blockSize : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"C         : "
        f"{ADAPTIVE_C}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    check_files()


    # -----------------------------------------------------
    # Count final test images
    # -----------------------------------------------------

    test_image_count = (
        count_test_images()
    )


    print(
        f"\nTest images: "
        f"{test_image_count}"
    )


    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    model, metrics = (
        run_test()
    )


    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    summary = extract_overall_metrics(
        metrics,
        test_image_count
    )


    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    save_overall_summary(
        summary
    )

    save_per_class_metrics(
        model,
        metrics
    )

    save_speed_metrics(
        metrics
    )


    # -----------------------------------------------------
    # Print final result
    # -----------------------------------------------------

    print_final_result(
        summary
    )


    print("\n" + "=" * 100)
    print("FINAL TEST COMPLETED - DO NOT RETUNE PARAMETERS")
    print("=" * 100)


    print(
        f"\nAll final results saved to:\n"
        f"{EVALUATION_DIR}"
    )


if __name__ == "__main__":
    main()