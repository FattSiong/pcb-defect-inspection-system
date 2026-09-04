from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

INPUT_CSV = (
    PROJECT_ROOT
    / "runs"
    / "student1_c"
    / "c_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "student1_c"
    / "analysis"
)

EXPECTED_C_VALUES = [
    2,
    5,
    8
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
        "C",
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

    df["C"] = pd.to_numeric(
        df["C"]
    ).astype(int)

    df = df[
        df["C"].isin(
            EXPECTED_C_VALUES
        )
    ].copy()

    df = df.sort_values(
        "C"
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
    print("ADAPTIVE GAUSSIAN C PARAMETER ANALYSIS")
    print("=" * 105)

    columns = [
        "C",
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
    print("BEST C VALUE")
    print("=" * 80)

    print(
        f"C         : {int(best['C'])}"
    )

    print(
        f"Precision : {best['Precision']:.4f}"
    )

    print(
        f"Recall    : {best['Recall']:.4f}"
    )

    print(
        f"F1-score  : {best['F1-score']:.4f}"
    )

    print(
        f"mAP50     : {best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95  : {best['mAP50-95']:.4f}"
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
        / "c_analysis.csv"
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
        df["C"],
        df["mAP50-95"],
        marker="o",
        linewidth=2
    )

    for _, row in df.iterrows():

        plt.annotate(
            f"{row['mAP50-95']:.4f}",
            (
                row["C"],
                row["mAP50-95"]
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    plt.title(
        "Validation mAP50-95 Across Different C Values"
    )

    plt.xlabel(
        "Adaptive Gaussian Thresholding C Value"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.xticks(
        EXPECTED_C_VALUES
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "c_map50_95.png"
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
            df["C"],
            df[metric],
            marker="o",
            label=metric
        )

    plt.title(
        "Validation Performance Across Different C Values"
    )

    plt.xlabel(
        "Adaptive Gaussian Thresholding C Value"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.xticks(
        EXPECTED_C_VALUES
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
        / "c_all_metrics.png"
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
    print("C PARAMETER ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()