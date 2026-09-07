"""Analyse and graph all three bilateral-morphology tuning stages."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# DEFAULT PATHS AND STAGE DEFINITIONS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_morphology_tuning"
)
DEFAULT_SEED = 42

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1_score",
    "map50",
    "map50_95",
]

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "map50": "mAP50",
    "map50_95": "mAP50-95",
}

COMMON_REQUIRED_COLUMNS = [
    "configuration",
    "seed",
    *METRIC_COLUMNS,
    "lowest_class_map50_95",
    "processing_seconds_per_image",
]

STAGES = {
    "stage1": {
        "name": "Stage 1 Bilateral Parameter",
        "short_name": "stage1_bilateral",
        "expected_configurations": ["B1", "B2", "B3"],
        "parameter_columns": [
            "bilateral_d",
            "sigma_color",
            "sigma_space",
        ],
        "map_title": (
            "Validation mAP50-95 Across Bilateral Configurations"
        ),
        "all_metrics_title": (
            "Validation Performance Across Bilateral Configurations"
        ),
        "x_label": "Bilateral Configuration",
    },
    "stage2": {
        "name": "Stage 2 Morphology Kernel",
        "short_name": "stage2_morphology",
        "expected_configurations": ["M1", "M2", "M3"],
        "parameter_columns": ["kernel_sizes"],
        "map_title": (
            "Validation mAP50-95 Across Morphology Kernel Sets"
        ),
        "all_metrics_title": (
            "Validation Performance Across Morphology Kernel Sets"
        ),
        "x_label": "Morphology Kernel Configuration",
    },
    "stage3": {
        "name": "Stage 3 Enhancement Weight",
        "short_name": "stage3_weights",
        "expected_configurations": ["W1", "W2", "W3"],
        "parameter_columns": ["alpha", "beta"],
        "map_title": (
            "Validation mAP50-95 Across Enhancement Weights"
        ),
        "all_metrics_title": (
            "Validation Performance Across Enhancement Weights"
        ),
        "x_label": "Enhancement Weight Configuration",
    },
}


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Create analysis tables and graphs for Stage 1 bilateral, "
            "Stage 2 morphology kernel, and Stage 3 enhancement-weight "
            "tuning results."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root containing the stage1, stage2, and stage3 result "
            f"directories. Default: {DEFAULT_RESULTS_ROOT}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Result seed to analyse. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output directory. The default is "
            "<results-root>/analysis/seed_<seed>."
        ),
    )
    return parser.parse_args()


# =========================================================
# LOAD AND VALIDATE RESULTS
# =========================================================

def stage_csv_path(results_root, stage_key, seed):
    return (
        results_root
        / stage_key
        / f"seed_{seed}"
        / f"{stage_key}_results.csv"
    )


def load_stage_results(results_root, stage_key, seed):
    stage = STAGES[stage_key]
    input_csv = stage_csv_path(results_root, stage_key, seed)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Cannot find {stage['name']} results:\n{input_csv}"
        )

    dataframe = pd.read_csv(input_csv)
    dataframe.columns = dataframe.columns.str.strip()

    required_columns = (
        COMMON_REQUIRED_COLUMNS
        + stage["parameter_columns"]
    )
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{stage['name']} is missing columns: {missing_columns}"
        )

    dataframe["configuration"] = (
        dataframe["configuration"].astype(str).str.strip()
    )
    expected = stage["expected_configurations"]
    dataframe = dataframe[
        dataframe["configuration"].isin(expected)
    ].copy()

    duplicate_configurations = (
        dataframe["configuration"]
        .value_counts()
        .loc[lambda counts: counts > 1]
        .index
        .tolist()
    )

    if duplicate_configurations:
        raise ValueError(
            f"{stage['name']} contains duplicate configurations: "
            f"{duplicate_configurations}"
        )

    found = set(dataframe["configuration"])
    missing_configurations = [
        configuration
        for configuration in expected
        if configuration not in found
    ]

    if missing_configurations:
        raise ValueError(
            f"{stage['name']} is missing configurations: "
            f"{missing_configurations}"
        )

    numeric_columns = [
        "seed",
        *METRIC_COLUMNS,
        "lowest_class_map50_95",
        "processing_seconds_per_image",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    unexpected_seeds = sorted(
        set(dataframe["seed"].astype(int)) - {seed}
    )

    if unexpected_seeds:
        raise ValueError(
            f"{stage['name']} contains rows from unexpected seeds: "
            f"{unexpected_seeds}"
        )

    order = {
        configuration: index
        for index, configuration in enumerate(expected)
    }
    dataframe["_configuration_order"] = (
        dataframe["configuration"].map(order)
    )
    dataframe = (
        dataframe
        .sort_values("_configuration_order")
        .drop(columns="_configuration_order")
        .reset_index(drop=True)
    )

    return dataframe, input_csv


# =========================================================
# ANALYSE RESULTS
# =========================================================

def analyse_results(dataframe):
    dataframe = dataframe.copy()
    dataframe["map50_95_change"] = dataframe["map50_95"].diff()
    dataframe["rank"] = (
        dataframe["map50_95"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return dataframe


def select_best_result(dataframe):
    ranked = dataframe.sort_values(
        by=[
            "map50_95",
            "lowest_class_map50_95",
            "f1_score",
            "processing_seconds_per_image",
        ],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def format_number(value):
    numeric_value = float(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:g}"


def configuration_tick_label(row, stage_key):
    configuration = row["configuration"]

    if stage_key == "stage1":
        return (
            f"{configuration}\n"
            f"d={format_number(row['bilateral_d'])}, "
            f"sigmaC={format_number(row['sigma_color'])}, "
            f"sigmaS={format_number(row['sigma_space'])}"
        )

    if stage_key == "stage2":
        kernels = str(row["kernel_sizes"]).replace(" ", "")
        return f"{configuration}\nkernels={kernels}"

    return (
        f"{configuration}\n"
        f"alpha={format_number(row['alpha'])}, "
        f"beta={format_number(row['beta'])}"
    )


# =========================================================
# PRINT AND SAVE ANALYSIS TABLES
# =========================================================

def print_results(dataframe, stage_key, input_csv):
    stage = STAGES[stage_key]
    display_columns = [
        "configuration",
        *stage["parameter_columns"],
        "precision",
        "recall",
        "f1_score",
        "map50",
        "map50_95",
        "lowest_class_map50_95",
        "map50_95_change",
        "rank",
    ]

    print("\n" + "=" * 120)
    print(f"{stage['name'].upper()} ANALYSIS")
    print("=" * 120)
    print(f"Input CSV: {input_csv}\n")
    print(
        dataframe[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best = select_best_result(dataframe)
    print("\n" + "=" * 80)
    print(f"BEST {stage['name'].upper()} CONFIGURATION")
    print("=" * 80)
    print(f"Configuration  : {best['configuration']}")

    if stage_key == "stage1":
        print(f"Bilateral d    : {int(best['bilateral_d'])}")
        print(f"Sigma colour   : {format_number(best['sigma_color'])}")
        print(f"Sigma space    : {format_number(best['sigma_space'])}")
    elif stage_key == "stage2":
        print(f"Kernel sizes   : {best['kernel_sizes']}")
    else:
        print(f"Alpha          : {format_number(best['alpha'])}")
        print(f"Beta           : {format_number(best['beta'])}")

    print(f"Precision      : {best['precision']:.4f}")
    print(f"Recall         : {best['recall']:.4f}")
    print(f"F1-score       : {best['f1_score']:.4f}")
    print(f"mAP50          : {best['map50']:.4f}")
    print(f"mAP50-95       : {best['map50_95']:.4f}")
    print(
        "Lowest class AP: "
        f"{best['lowest_class_map50_95']:.4f}"
    )


def save_analysis(dataframe, stage_key, output_dir):
    stage = STAGES[stage_key]
    output_path = (
        output_dir
        / f"{stage['short_name']}_analysis.csv"
    )
    dataframe.to_csv(output_path, index=False)
    print(f"\nAnalysis CSV saved:\n{output_path}")
    return output_path


# =========================================================
# GRAPH 1 - mAP50-95
# =========================================================

def plot_map5095(dataframe, stage_key, output_dir):
    stage = STAGES[stage_key]
    x_positions = np.arange(len(dataframe))
    tick_labels = [
        configuration_tick_label(row, stage_key)
        for _, row in dataframe.iterrows()
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_positions,
        dataframe["map50_95"],
        marker="o",
        markersize=8,
        linewidth=2,
    )

    best_index = int(dataframe["map50_95"].idxmax())
    plt.scatter(
        [best_index],
        [dataframe.loc[best_index, "map50_95"]],
        color="red",
        s=90,
        zorder=3,
        label="Highest mAP50-95",
    )

    for index, row in dataframe.iterrows():
        plt.annotate(
            f"{row['map50_95']:.4f}",
            (index, row["map50_95"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title(stage["map_title"])
    plt.xlabel(stage["x_label"])
    plt.ylabel("Validation mAP50-95")
    plt.xticks(x_positions, tick_labels)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        output_dir
        / f"{stage['short_name']}_map50_95.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"mAP50-95 graph saved:\n{output_path}")
    return output_path


# =========================================================
# GRAPH 2 - ALL METRICS
# =========================================================

def plot_all_metrics(dataframe, stage_key, output_dir):
    stage = STAGES[stage_key]
    x_positions = np.arange(len(dataframe))
    tick_labels = [
        configuration_tick_label(row, stage_key)
        for _, row in dataframe.iterrows()
    ]

    plt.figure(figsize=(11, 6))

    for metric in METRIC_COLUMNS:
        plt.plot(
            x_positions,
            dataframe[metric],
            marker="o",
            label=METRIC_LABELS[metric],
        )

    plt.title(stage["all_metrics_title"])
    plt.xlabel(stage["x_label"])
    plt.ylabel("Performance Score")
    plt.xticks(x_positions, tick_labels)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        output_dir
        / f"{stage['short_name']}_all_metrics.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"All-metrics graph saved:\n{output_path}")
    return output_path


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_arguments()
    results_root = args.results_root.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_root / "analysis" / f"seed_{args.seed}"
    )

    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found:\n{results_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BILATERAL + MULTI-SCALE MORPHOLOGY TUNING ANALYSIS")
    print("=" * 80)
    print(f"Results root : {results_root}")
    print(f"Seed         : {args.seed}")
    print(f"Output       : {output_dir}")

    for stage_key in STAGES:
        dataframe, input_csv = load_stage_results(
            results_root,
            stage_key,
            args.seed,
        )
        dataframe = analyse_results(dataframe)
        print_results(dataframe, stage_key, input_csv)
        save_analysis(dataframe, stage_key, output_dir)
        plot_map5095(dataframe, stage_key, output_dir)
        plot_all_metrics(dataframe, stage_key, output_dir)

    print("\n" + "=" * 80)
    print("ALL THREE TUNING-STAGE ANALYSES COMPLETED")
    print("=" * 80)
    print(f"All analysis files:\n{output_dir}")


if __name__ == "__main__":
    main()
