from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
)

# Complete 50-150, 100-200, 150-250 comparison
INPUT_CSV = (
    PROJECT_ROOT
    / "runs"
    / "student2_canny_thresh"
    / "canny_thresh_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "student2_canny_thresh"
    / "analysis"
)

EXPECTED_THRESHOLDS = [
    "50-150",
    "100-200",
    "150-250"
]


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    required_columns = [
        "Canny Threshold",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            f"Missing columns: {missing_columns}"
        )

    df["Canny Threshold"] = (
        df["Canny Threshold"]
        .astype(str)
    )

    df = df[
        df["Canny Threshold"].isin(
            EXPECTED_THRESHOLDS
        )
    ].copy()

    df["Canny Threshold"] = pd.Categorical(
        df["Canny Threshold"],
        categories=EXPECTED_THRESHOLDS,
        ordered=True
    )

    df = df.sort_values(
        "Canny Threshold"
    ).reset_index(drop=True)

    df["Canny Threshold"] = (
        df["Canny Threshold"]
        .astype(str)
    )

    found = set(
        df["Canny Threshold"].tolist()
    )

    missing = [
        value
        for value in EXPECTED_THRESHOLDS
        if value not in found
    ]

    if missing:

        raise ValueError(
            f"Missing Canny threshold results: {missing}"
        )

    return df


# =========================================================
# ANALYSE
# =========================================================

def analyse_results(df):

    df = df.copy()


    df["mAP50-95 Change"] = (
        df["mAP50-95"].diff()
    )


    df["Rank"] = (
        df["mAP50-95"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    return df


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(df):

    print("\n" + "=" * 105)
    print("CANNY THRESHOLD VALIDATION ANALYSIS")
    print("=" * 105)

    columns = [
        "Canny Threshold",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
        "mAP50-95 Change",
        "Rank"
    ]

    print(
        df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best = df.loc[
        df["mAP50-95"].idxmax()
    ]

    print("\n" + "=" * 80)
    print("BEST CANNY THRESHOLD")
    print("=" * 80)

    print(
        f"Canny Threshold : {best['Canny Threshold']}"
    )

    print(
        f"Precision       : {best['Precision']:.4f}"
    )

    print(
        f"Recall          : {best['Recall']:.4f}"
    )

    print(
        f"F1-score        : {best['F1-score']:.4f}"
    )

    print(
        f"mAP50           : {best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95        : {best['mAP50-95']:.4f}"
    )


# =========================================================
# SAVE ANALYSIS TABLE
# =========================================================

def save_analysis(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "canny_thresh_analysis.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nAnalysis CSV saved:\n{output_path}"
    )


# =========================================================
# GRAPH 1 - mAP50-95
# =========================================================

def plot_map5095(df):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        df["Canny Threshold"],
        df["mAP50-95"],
        marker="o",
        linewidth=2
    )

    for _, row in df.iterrows():

        plt.annotate(
            f"{row['mAP50-95']:.4f}",
            (
                row["Canny Threshold"],
                row["mAP50-95"]
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    plt.title(
        "Validation mAP50-95 Across Different Canny Thresholds"
    )

    plt.xlabel(
        "Canny Threshold (Lower-Upper)"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "canny_thresh_map50_95.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"mAP50-95 graph saved:\n{output_path}"
    )


# =========================================================
# GRAPH 2 - ALL METRICS
# =========================================================

def plot_all_metrics(df):

    plt.figure(
        figsize=(10, 6)
    )

    metric_columns = [
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    for metric in metric_columns:

        plt.plot(
            df["Canny Threshold"],
            df[metric],
            marker="o",
            label=metric
        )

    plt.title(
        "Validation Performance Across Different Canny Thresholds"
    )

    plt.xlabel(
        "Canny Threshold (Lower-Upper)"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "canny_thresh_all_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"All-metrics graph saved:\n{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_results()

    df = analyse_results(
        df
    )

    print_results(
        df
    )

    save_analysis(
        df
    )

    plot_map5095(
        df
    )

    plot_all_metrics(
        df
    )

    print("\n" + "=" * 80)
    print("CANNY THRESHOLD ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()