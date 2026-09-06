from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "final_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

STUDENTS = {
    "A": {
        "overall":
            RESULTS_DIR
            / "student1_overall_test_metrics.csv",

        "per_class":
            RESULTS_DIR
            / "student1_all_classes_summary.csv",

        "technique":
            "CLAHE + Adaptive Gaussian Thresholding",
    },

    "B": {
        "overall":
            RESULTS_DIR
            / "student2_overall_test_metrics.csv",

        "per_class":
            RESULTS_DIR
            / "student2_all_classes_summary.csv",

        "technique":
            "Canny Edge Detection + Hough Transform",
    },

    "C": {
        "overall":
            RESULTS_DIR
            / "student3_overall_test_metrics.csv",

        "per_class":
            RESULTS_DIR
            / "student3_all_classes_summary.csv",

        "technique":
            "Gaussian Filtering + Uniform LBP Luminance Fusion",
    },

    "D": {
        "overall":
            RESULTS_DIR
            / "student4_overall_test_metrics.csv",

        "per_class":
            RESULTS_DIR
            / "student4_all_classes_summary.csv",

        "technique":
            (
                "Bilateral Filtering + "
                "Multi-scale Top-Hat/Black-Hat Morphology"
            ),
    },
}


# ============================================================
# METRICS
# ============================================================

METRICS = [
    "Precision",
    "Recall",
    "F1-score",
    "mAP50",
    "mAP50-95",
]

PRIMARY_METRIC = "mAP50-95"


# ============================================================
# DEFECT DISPLAY NAMES
# ============================================================

DEFECT_DISPLAY_NAMES = {
    "mouse_bite": "Mouse Bite",
    "spur": "Spur",
    "missing_hole": "Missing Hole",
    "short": "Short",
    "open_circuit": "Open Circuit",
    "spurious_copper": "Spurious Copper",
}


# ============================================================
# CHECK INPUT FILES
# ============================================================

def check_files():

    missing_files = []

    for technique_id, info in STUDENTS.items():

        if not info["overall"].exists():
            missing_files.append(
                str(info["overall"])
            )

        if not info["per_class"].exists():
            missing_files.append(
                str(info["per_class"])
            )

    if missing_files:

        print("\nMissing files:")

        for path in missing_files:
            print(path)

        raise FileNotFoundError(
            "One or more required CSV files are missing."
        )


# ============================================================
# READ OVERALL RESULTS
# ============================================================

def read_overall_results():

    rows = []

    for technique_id, info in STUDENTS.items():

        df = pd.read_csv(
            info["overall"]
        )

        if df.empty:

            raise ValueError(
                f"Overall CSV for Technique "
                f"{technique_id} is empty."
            )

        row = df.iloc[0]

        result = {
            "Technique ID":
                technique_id,

            "Technique":
                info["technique"],
        }

        for metric in METRICS:

            if metric not in df.columns:

                raise ValueError(
                    f"{metric} missing from "
                    f"{info['overall'].name}"
                )

            result[metric] = float(
                row[metric]
            )

        rows.append(
            result
        )

    overall_df = pd.DataFrame(
        rows
    )

    return overall_df


# ============================================================
# READ PER-CLASS RESULTS
# ============================================================

def read_per_class_results():

    rows = []

    for technique_id, info in STUDENTS.items():

        df = pd.read_csv(
            info["per_class"]
        )

        required_columns = [
            "Class ID",
            "Defect Class",
            *METRICS,
        ]

        for column in required_columns:

            if column not in df.columns:

                raise ValueError(
                    f"{column} missing from "
                    f"{info['per_class'].name}"
                )

        for _, row in df.iterrows():

            defect_name = str(
                row["Defect Class"]
            )

            result = {
                "Technique ID":
                    technique_id,

                "Technique":
                    info["technique"],

                "Class ID":
                    int(
                        row["Class ID"]
                    ),

                "Defect Class":
                    defect_name,

                "Display Name":
                    DEFECT_DISPLAY_NAMES.get(
                        defect_name,
                        defect_name
                        .replace("_", " ")
                        .title()
                    ),
            }

            for metric in METRICS:

                result[metric] = float(
                    row[metric]
                )

            rows.append(
                result
            )

    per_class_df = pd.DataFrame(
        rows
    )

    return per_class_df


# ============================================================
# SAVE OVERALL COMPARISON CSV
# ============================================================

def save_overall_comparison(
    overall_df
):

    output_path = (
        OUTPUT_DIR
        / "overall_comparison.csv"
    )

    overall_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# SAVE PER-CLASS mAP50-95 COMPARISON CSV
# ============================================================

def create_per_class_map_table(
    per_class_df
):

    pivot = (
        per_class_df
        .pivot(
            index="Display Name",
            columns="Technique ID",
            values=PRIMARY_METRIC
        )
    )

    # Keep original class order
    class_order = [
        "Mouse Bite",
        "Spur",
        "Missing Hole",
        "Short",
        "Open Circuit",
        "Spurious Copper",
    ]

    pivot = pivot.reindex(
        class_order
    )

    # Rename columns
    pivot.columns = [
        f"Technique {column}"
        for column in pivot.columns
    ]

    # Determine the best technique per class
    technique_columns = [
        column
        for column in pivot.columns
        if column.startswith(
            "Technique "
        )
    ]

    pivot["Best Technique"] = (
        pivot[
            technique_columns
        ]
        .idxmax(
            axis=1
        )
    )

    pivot["Best mAP50-95"] = (
        pivot[
            technique_columns
        ]
        .max(
            axis=1
        )
    )

    pivot = pivot.reset_index()

    output_path = (
        OUTPUT_DIR
        / "per_class_map50_95_comparison.csv"
    )

    pivot.to_csv(
        output_path,
        index=False
    )

    return pivot, output_path


# ============================================================
# SAVE BEST TECHNIQUE BY DEFECT CSV
# ============================================================

def save_best_technique_by_defect(
    comparison_df
):

    columns = [
        "Display Name",
        "Best Technique",
        "Best mAP50-95",
    ]

    best_df = (
        comparison_df[
            columns
        ]
        .copy()
    )

    best_df.columns = [
        "Defect Class",
        "Best Technique",
        "Best mAP50-95",
    ]

    output_path = (
        OUTPUT_DIR
        / "best_technique_by_defect.csv"
    )

    best_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# GRAPH 1
# OVERALL ALL-METRICS COMPARISON
# ============================================================

def plot_overall_all_metrics(
    overall_df
):

    technique_ids = (
        overall_df[
            "Technique ID"
        ]
        .tolist()
    )

    x = np.arange(
        len(
            technique_ids
        )
    )

    number_of_metrics = len(
        METRICS
    )

    width = 0.15

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    for metric_index, metric in enumerate(
        METRICS
    ):

        offset = (
            metric_index
            - (
                number_of_metrics
                - 1
            ) / 2
        ) * width

        values = (
            overall_df[
                metric
            ]
            .to_numpy()
        )

        bars = axis.bar(
            x + offset,
            values,
            width,
            label=metric
        )

        # Show values above bars
        for bar, value in zip(
            bars,
            values
        ):

            axis.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height()
                + 0.008,

                f"{value:.3f}",

                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    axis.set_title(
        "Overall Test Performance Comparison "
        "of Four Techniques",
        fontsize=16
    )

    axis.set_xlabel(
        "Image-Processing Technique",
        fontsize=12
    )

    axis.set_ylabel(
        "Score",
        fontsize=12
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            f"Technique {technique_id}"
            for technique_id
            in technique_ids
        ]
    )

    axis.set_ylim(
        0,
        1.05
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    axis.legend(
        title="Metric",
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "overall_all_metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# GRAPH 2
# OVERALL mAP50-95 COMPARISON
# ============================================================

def plot_overall_map50_95(
    overall_df
):

    labels = [
        f"Technique {value}"
        for value in overall_df[
            "Technique ID"
        ]
    ]

    values = (
        overall_df[
            PRIMARY_METRIC
        ]
        .to_numpy()
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    bars = axis.bar(
        labels,
        values
    )

    for bar, value in zip(
        bars,
        values
    ):

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.008,

            f"{value:.4f}",

            ha="center",
            va="bottom",
            fontsize=11
        )

    axis.set_title(
        "Final Test mAP50-95 Comparison "
        "of Four Techniques",
        fontsize=16
    )

    axis.set_xlabel(
        "Image-Processing Technique",
        fontsize=12
    )

    axis.set_ylabel(
        "mAP50-95",
        fontsize=12
    )

    axis.set_ylim(
        0,
        0.6
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "overall_map50_95_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# GRAPH 3
# PER-CLASS mAP50-95 COMPARISON
# ============================================================

def plot_per_class_map50_95(
    comparison_df
):

    defect_classes = (
        comparison_df[
            "Display Name"
        ]
        .tolist()
    )

    technique_columns = [
        "Technique A",
        "Technique B",
        "Technique C",
        "Technique D",
    ]

    x = np.arange(
        len(
            defect_classes
        )
    )

    width = 0.19

    figure, axis = plt.subplots(
        figsize=(15, 8)
    )

    for index, technique in enumerate(
        technique_columns
    ):

        offset = (
            index
            - (
                len(
                    technique_columns
                )
                - 1
            ) / 2
        ) * width

        values = (
            comparison_df[
                technique
            ]
            .to_numpy()
        )

        bars = axis.bar(
            x + offset,
            values,
            width,
            label=technique
        )

        for bar, value in zip(
            bars,
            values
        ):

            axis.text(
                bar.get_x()
                + bar.get_width() / 2,

                value + 0.006,

                f"{value:.3f}",

                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    axis.set_title(
        "Per-Class Test mAP50-95 Comparison "
        "Across Four Techniques",
        fontsize=16
    )

    axis.set_xlabel(
        "PCB Defect Class",
        fontsize=12
    )

    axis.set_ylabel(
        "mAP50-95",
        fontsize=12
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        defect_classes,
        rotation=20,
        ha="right"
    )

    axis.set_ylim(
        0,
        0.65
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    axis.legend(
        title="Technique",
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "per_class_map50_95_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# GRAPH 4
# BEST TECHNIQUE FOR EACH DEFECT CLASS
# ============================================================

def plot_best_technique_by_defect(
    comparison_df
):

    defect_classes = (
        comparison_df[
            "Display Name"
        ]
        .tolist()
    )

    best_values = (
        comparison_df[
            "Best mAP50-95"
        ]
        .to_numpy()
    )

    best_techniques = (
        comparison_df[
            "Best Technique"
        ]
        .tolist()
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    bars = axis.bar(
        defect_classes,
        best_values
    )

    for (
        bar,
        value,
        technique
    ) in zip(
        bars,
        best_values,
        best_techniques
    ):

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.007,

            (
                f"{technique}\n"
                f"{value:.4f}"
            ),

            ha="center",
            va="bottom",
            fontsize=9
        )

    axis.set_title(
        "Best-Performing Technique "
        "for Each PCB Defect Class",
        fontsize=16
    )

    axis.set_xlabel(
        "PCB Defect Class",
        fontsize=12
    )

    axis.set_ylabel(
        "Best mAP50-95",
        fontsize=12
    )

    axis.set_ylim(
        0,
        0.65
    )

    axis.set_xticklabels(
        defect_classes,
        rotation=20,
        ha="right"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "best_technique_by_defect.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# PRINT OVERALL RESULT SUMMARY
# ============================================================

def print_overall_summary(
    overall_df
):

    print()
    print("=" * 90)
    print(
        "FINAL OVERALL TEST COMPARISON"
    )
    print("=" * 90)

    display_df = (
        overall_df[
            [
                "Technique ID",
                *METRICS,
            ]
        ]
        .copy()
    )

    display_df[
        METRICS
    ] = (
        display_df[
            METRICS
        ]
        .round(
            4
        )
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    best_row_index = (
        overall_df[
            PRIMARY_METRIC
        ]
        .idxmax()
    )

    best_row = (
        overall_df.loc[
            best_row_index
        ]
    )

    print()
    print(
        "Best overall technique "
        "based on mAP50-95:"
    )

    print(
        f"Technique "
        f"{best_row['Technique ID']}"
    )

    print(
        best_row[
            "Technique"
        ]
    )

    print(
        f"mAP50-95 = "
        f"{best_row[PRIMARY_METRIC]:.4f}"
    )


# ============================================================
# PRINT PER-CLASS SUMMARY
# ============================================================

def print_per_class_summary(
    comparison_df
):

    print()
    print("=" * 90)
    print(
        "PER-CLASS mAP50-95 COMPARISON"
    )
    print("=" * 90)

    display_df = (
        comparison_df.copy()
    )

    numeric_columns = [
        "Technique A",
        "Technique B",
        "Technique C",
        "Technique D",
        "Best mAP50-95",
    ]

    display_df[
        numeric_columns
    ] = (
        display_df[
            numeric_columns
        ]
        .round(
            4
        )
    )

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT TECHNIQUE MAPPING
# ============================================================

def print_technique_mapping():

    print()
    print("=" * 90)
    print(
        "TECHNIQUE MAPPING"
    )
    print("=" * 90)

    for technique_id, info in (
        STUDENTS.items()
    ):

        print(
            f"Technique {technique_id}: "
            f"{info['technique']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(
        "FINAL FOUR-TECHNIQUE "
        "COMPARISON ANALYSIS"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    check_files()


    # --------------------------------------------------------
    # Read results
    # --------------------------------------------------------

    overall_df = (
        read_overall_results()
    )

    per_class_df = (
        read_per_class_results()
    )


    # --------------------------------------------------------
    # Save comparison tables
    # --------------------------------------------------------

    overall_csv = (
        save_overall_comparison(
            overall_df
        )
    )

    (
        per_class_comparison_df,
        per_class_csv
    ) = create_per_class_map_table(
        per_class_df
    )

    best_by_defect_csv = (
        save_best_technique_by_defect(
            per_class_comparison_df
        )
    )


    # --------------------------------------------------------
    # Generate graphs
    # --------------------------------------------------------

    overall_metrics_graph = (
        plot_overall_all_metrics(
            overall_df
        )
    )

    overall_map_graph = (
        plot_overall_map50_95(
            overall_df
        )
    )

    per_class_graph = (
        plot_per_class_map50_95(
            per_class_comparison_df
        )
    )

    best_by_defect_graph = (
        plot_best_technique_by_defect(
            per_class_comparison_df
        )
    )


    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_technique_mapping()

    print_overall_summary(
        overall_df
    )

    print_per_class_summary(
        per_class_comparison_df
    )


    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "FILES GENERATED"
    )
    print("=" * 90)

    print()
    print(
        "Tables:"
    )

    print(
        f"  {overall_csv}"
    )

    print(
        f"  {per_class_csv}"
    )

    print(
        f"  {best_by_defect_csv}"
    )

    print()
    print(
        "Graphs:"
    )

    print(
        f"  {overall_metrics_graph}"
    )

    print(
        f"  {overall_map_graph}"
    )

    print(
        f"  {per_class_graph}"
    )

    print(
        f"  {best_by_defect_graph}"
    )

    print()
    print(
        f"All files saved under:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()