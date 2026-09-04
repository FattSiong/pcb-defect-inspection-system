from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

# ---------------------------------------------------------
# ORIGINAL REDUCED DATASET 100-EPOCH MODEL
# ---------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "epoch_testing"
    / "reduced_dataset_100_epochs"
    / "weights"
    / "best.pt"
)

# Original reduced dataset
# No CLAHE / thresholding / other preprocessing
DATA_YAML = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced\data.yaml"
)

# Output folders
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "original_reduced_final_test"
)

PER_CLASS_DIR = (
    OUTPUT_ROOT
    / "per_class"
)

GRAPH_DIR = (
    OUTPUT_ROOT
    / "graphs"
)


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# =========================================================
# GRAPH FUNCTIONS
# =========================================================

def generate_graphs(
    per_class_df,
    precision,
    recall,
    f1,
    map50,
    map50_95
):

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # GRAPH 1: OVERALL PERFORMANCE
    # -----------------------------------------------------

    metric_names = [
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    metric_values = [
        precision,
        recall,
        f1,
        map50,
        map50_95
    ]

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.title(
        "Overall Test Performance "
        "of Original Image Baseline"
    )

    plt.xlabel(
        "Evaluation Metric"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "overall_test_performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # -----------------------------------------------------
    # GRAPH 2: PER-CLASS mAP50-95
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        per_class_df[
            "Defect Class"
        ],
        per_class_df[
            "mAP50-95"
        ]
    )

    plt.title(
        "Per-Class Test mAP50-95 "
        "of Original Image Baseline"
    )

    plt.xlabel(
        "PCB Defect Class"
    )

    plt.ylabel(
        "mAP50-95"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    for bar, value in zip(
        bars,
        per_class_df[
            "mAP50-95"
        ]
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "per_class_map50_95.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # -----------------------------------------------------
    # GRAPH 3: ALL PER-CLASS METRICS
    # -----------------------------------------------------

    metrics_to_plot = [
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    ax = (
        per_class_df
        .set_index(
            "Defect Class"
        )[
            metrics_to_plot
        ]
        .plot(
            kind="bar",
            figsize=(12, 7)
        )
    )

    plt.title(
        "Per-Class Test Performance "
        "of Original Image Baseline"
    )

    plt.xlabel(
        "PCB Defect Class"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.legend(
        title="Metric",
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "per_class_all_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # CREATE FOLDERS
    # -----------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    PER_CLASS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # VERIFY FILES
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n100-epoch baseline model "
            "was not found:\n"
            f"{MODEL_PATH}\n"
        )

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            "\nDataset YAML was not found:\n"
            f"{DATA_YAML}\n"
        )


    # -----------------------------------------------------
    # TEST INFORMATION
    # -----------------------------------------------------

    print("=" * 70)
    print(
        "ORIGINAL REDUCED DATASET "
        "- FINAL BASELINE TEST"
    )
    print("=" * 70)

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Dataset : {DATA_YAML}"
    )

    print("\nBaseline Configuration")
    print("-" * 70)

    print(
        "Image Processing      : None"
    )

    print(
        "Training Dataset      : "
        "Reduced Original Images"
    )

    print(
        "Training Images       : 1,200"
    )

    print(
        "Training Epoch Limit  : 100"
    )

    print(
        "Model                 : YOLOv8n"
    )

    print(
        "Image Size            : 640 x 640"
    )

    print(
        "Test Batch Size       : 4"
    )


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print(
        "\nLoading original-image "
        "baseline model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )


    # -----------------------------------------------------
    # RUN FINAL TEST
    # -----------------------------------------------------

    print(
        "\nRunning final baseline test..."
    )

    print(
        "-" * 70
    )

    start_time = (
        time.perf_counter()
    )

    metrics = model.val(
        data=str(DATA_YAML),

        split="test",

        imgsz=640,

        batch=4,

        device=0,

        # Windows-safe
        workers=0,

        plots=True,

        save_json=False,

        project=str(
            OUTPUT_ROOT
        ),

        name="yolo_results",

        exist_ok=True
    )

    end_time = (
        time.perf_counter()
    )

    total_test_time = (
        end_time
        - start_time
    )


    # =====================================================
    # OVERALL METRICS
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


    if precision + recall > 0:

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


    # -----------------------------------------------------
    # YOLO SPEED
    # -----------------------------------------------------

    speed = metrics.speed

    yolo_preprocess = float(
        speed.get(
            "preprocess",
            0.0
        )
    )

    inference_time = float(
        speed.get(
            "inference",
            0.0
        )
    )

    postprocess_time = float(
        speed.get(
            "postprocess",
            0.0
        )
    )

    total_yolo_time = (
        yolo_preprocess
        + inference_time
        + postprocess_time
    )


    # =====================================================
    # SAVE OVERALL RESULT
    # =====================================================

    overall_data = {

        "Technique":
            "Original Image Baseline",

        "Model":
            "YOLOv8n",

        "Image Processing":
            "None",

        "Training Epoch Limit":
            100,

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

        "Classical Image Processing Time (ms/image)":
            0.0,

        "YOLO Preprocess Time (ms/image)":
            yolo_preprocess,

        "Inference Time (ms/image)":
            inference_time,

        "Postprocess Time (ms/image)":
            postprocess_time,

        "Total YOLO Time (ms/image)":
            total_yolo_time,

        "Total Test Wall Time (seconds)":
            total_test_time,
    }


    overall_df = pd.DataFrame(
        [overall_data]
    )

    overall_csv = (
        OUTPUT_ROOT
        / "overall_test_metrics.csv"
    )

    overall_df.to_csv(
        overall_csv,
        index=False
    )


    # -----------------------------------------------------
    # JSON RESULT
    # -----------------------------------------------------

    overall_json = (
        OUTPUT_ROOT
        / "overall_test_metrics.json"
    )

    with open(
        overall_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            overall_data,
            f,
            indent=4
        )


    # =====================================================
    # PER-CLASS METRICS
    # =====================================================

    precision_per_class = (
        metrics.box.p
    )

    recall_per_class = (
        metrics.box.r
    )

    ap50_per_class = (
        metrics.box.ap50
    )

    ap_per_class = (
        metrics.box.ap
    )

    per_class_rows = []


    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        p = float(
            precision_per_class[
                class_id
            ]
        )

        r = float(
            recall_per_class[
                class_id
            ]
        )

        ap50 = float(
            ap50_per_class[
                class_id
            ]
        )

        map_50_95 = float(
            ap_per_class[
                class_id
            ]
        )


        if p + r > 0:

            class_f1 = (
                2
                * p
                * r
                / (
                    p + r
                )
            )

        else:

            class_f1 = 0.0


        row = {

            "Class ID":
                class_id,

            "Defect Class":
                class_name,

            "Precision":
                p,

            "Recall":
                r,

            "F1-score":
                class_f1,

            "mAP50":
                ap50,

            "mAP50-95":
                map_50_95,
        }

        per_class_rows.append(
            row
        )


        # ---------------------------------------------
        # SAVE INDIVIDUAL DEFECT CSV
        # ---------------------------------------------

        class_df = pd.DataFrame(
            [row]
        )

        class_output = (
            PER_CLASS_DIR
            / f"{class_name}.csv"
        )

        class_df.to_csv(
            class_output,
            index=False
        )


    # =====================================================
    # COMBINED PER-CLASS RESULT
    # =====================================================

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    all_classes_csv = (
        OUTPUT_ROOT
        / "all_classes_summary.csv"
    )

    per_class_df.to_csv(
        all_classes_csv,
        index=False
    )


    # =====================================================
    # GENERATE GRAPHS
    # =====================================================

    print(
        "\nGenerating baseline graphs..."
    )

    generate_graphs(
        per_class_df,
        precision,
        recall,
        f1,
        map50,
        map50_95
    )


    # =====================================================
    # DISPLAY OVERALL RESULT
    # =====================================================

    print("\n")
    print("=" * 70)
    print(
        "OVERALL ORIGINAL-IMAGE "
        "BASELINE RESULT"
    )
    print("=" * 70)

    print(
        f"Precision    : "
        f"{precision:.4f}"
    )

    print(
        f"Recall       : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score     : "
        f"{f1:.4f}"
    )

    print(
        f"mAP50        : "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95     : "
        f"{map50_95:.4f}"
    )


    print(
        "\nYOLO Speed"
    )

    print(
        f"Preprocess   : "
        f"{yolo_preprocess:.3f} ms/image"
    )

    print(
        f"Inference    : "
        f"{inference_time:.3f} ms/image"
    )

    print(
        f"Postprocess  : "
        f"{postprocess_time:.3f} ms/image"
    )

    print(
        f"Total YOLO   : "
        f"{total_yolo_time:.3f} ms/image"
    )

    print(
        f"Wall Time    : "
        f"{total_test_time:.2f} seconds"
    )


    # =====================================================
    # DISPLAY PER-CLASS RESULT
    # =====================================================

    print("\n")
    print("=" * 70)
    print(
        "PER-CLASS BASELINE TEST RESULTS"
    )
    print("=" * 70)

    display_columns = [
        "Defect Class",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    print(
        per_class_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    # =====================================================
    # FILE LOCATIONS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)


    print("\nOverall result:")

    print(
        f"  {overall_csv}"
    )

    print(
        f"  {overall_json}"
    )


    print("\nAll defect classes:")

    print(
        f"  {all_classes_csv}"
    )


    print(
        "\nIndividual defect results:"
    )

    for class_name in (
        CLASS_NAMES.values()
    ):

        print(
            f"  "
            f"{PER_CLASS_DIR / f'{class_name}.csv'}"
        )


    print("\nGraphs:")

    print(
        f"  "
        f"{GRAPH_DIR / 'overall_test_performance.png'}"
    )

    print(
        f"  "
        f"{GRAPH_DIR / 'per_class_map50_95.png'}"
    )

    print(
        f"  "
        f"{GRAPH_DIR / 'per_class_all_metrics.png'}"
    )


    print("\nYOLO plots:")

    print(
        f"  "
        f"{OUTPUT_ROOT / 'yolo_results'}"
    )


    print("\n")
    print("=" * 70)
    print(
        "ORIGINAL REDUCED DATASET "
        "FINAL TEST COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


# =========================================================
# WINDOWS MULTIPROCESSING PROTECTION
# =========================================================

if __name__ == "__main__":
    main()