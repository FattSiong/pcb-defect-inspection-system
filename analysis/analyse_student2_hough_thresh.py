from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
)

INPUT_CSV = (
    PROJECT_ROOT
    / "runs"
    / "student2_hough_thresh"
    / "hough_thresh_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "student2_hough_thresh"
    / "analysis"
)

EXPECTED_HOUGH_VALUES = [
    30,
    50,
    80
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
        "Hough Threshold",
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

    df["Hough Threshold"] = pd.to_numeric(
        df["Hough Threshold"]
    ).astype(int)

    df = df[
        df["Hough Threshold"].isin(
            EXPECTED_HOUGH_VALUES
        )
    ].copy()

    df = df.sort_values(
        "Hough Threshold"
    ).reset_index(drop=True)

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
    print("HOUGH THRESHOLD PARAMETER ANALYSIS")
    print("=" * 105)

    columns = [
        "Hough Threshold",
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
    print("BEST HOUGH THRESHOLD")
    print("=" * 80)

    print(
        f"Hough Threshold : {int(best['Hough Threshold'])}"
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
# SAVE TABLE
# =========================================================

def save_analysis(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "hough_thresh_analysis.csv"
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
        figsize=(8, 6)
    )

    plt.plot(
        df["Hough Threshold"],
        df["mAP50-95"],
        marker="o",
        linewidth=2
    )

    for _, row in df.iterrows():

        plt.annotate(
            f"{row['mAP50-95']:.4f}",
            (
                row["Hough Threshold"],
                row["mAP50-95"]
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    plt.title(
        "Validation mAP50-95 Across Different Hough Thresholds"
    )

    plt.xlabel(
        "Hough Voting Threshold"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.xticks(
        EXPECTED_HOUGH_VALUES
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "hough_thresh_map50_95.png"
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
        figsize=(9, 6)
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
            df["Hough Threshold"],
            df[metric],
            marker="o",
            label=metric
        )

    plt.title(
        "Validation Performance Across Different Hough Thresholds"
    )

    plt.xlabel(
        "Hough Voting Threshold"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.xticks(
        EXPECTED_HOUGH_VALUES
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
        / "hough_thresh_all_metrics.png"
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
    print("HOUGH THRESHOLD ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()