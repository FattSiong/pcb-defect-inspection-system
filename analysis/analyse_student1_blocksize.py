from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

# Complete 11, 21, 31, 41, 51 comparison
INPUT_CSV = (
    PROJECT_ROOT
    / "runs"
    / "student1_blocksize"
    / "blocksize_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "student1_blocksize"
    / "analysis"
)

EXPECTED_BLOCK_SIZES = [
    11,
    21,
    31,
    41,
    51
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
        "Block Size",
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

    df["Block Size"] = pd.to_numeric(
        df["Block Size"]
    ).astype(int)

    df = df[
        df["Block Size"].isin(
            EXPECTED_BLOCK_SIZES
        )
    ].copy()

    df = df.sort_values(
        "Block Size"
    ).reset_index(drop=True)

    found = set(
        df["Block Size"].tolist()
    )

    missing = [
        value
        for value in EXPECTED_BLOCK_SIZES
        if value not in found
    ]

    if missing:

        raise ValueError(
            f"Missing block-size results: {missing}"
        )

    return df


# =========================================================
# ANALYSE
# =========================================================

def analyse_results(df):

    df = df.copy()

    # Change compared with previous tested block size
    df["mAP50-95 Change"] = (
        df["mAP50-95"].diff()
    )

    # Rank according to primary metric
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
    print("BLOCK SIZE VALIDATION ANALYSIS")
    print("=" * 105)

    columns = [
        "Block Size",
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
    print("BEST BLOCK SIZE")
    print("=" * 80)

    print(
        f"Block Size : {int(best['Block Size'])}"
    )

    print(
        f"Precision  : {best['Precision']:.4f}"
    )

    print(
        f"Recall     : {best['Recall']:.4f}"
    )

    print(
        f"F1-score   : {best['F1-score']:.4f}"
    )

    print(
        f"mAP50      : {best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95   : {best['mAP50-95']:.4f}"
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
        / "blocksize_analysis.csv"
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
        df["Block Size"],
        df["mAP50-95"],
        marker="o",
        linewidth=2
    )

    for _, row in df.iterrows():

        plt.annotate(
            f"{row['mAP50-95']:.4f}",
            (
                row["Block Size"],
                row["mAP50-95"]
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    plt.title(
        "Validation mAP50-95 Across Different Block Sizes"
    )

    plt.xlabel(
        "Adaptive Gaussian Thresholding Block Size"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.xticks(
        EXPECTED_BLOCK_SIZES
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "blocksize_map50_95.png"
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
            df["Block Size"],
            df[metric],
            marker="o",
            label=metric
        )

    plt.title(
        "Validation Performance Across Different Block Sizes"
    )

    plt.xlabel(
        "Adaptive Gaussian Thresholding Block Size"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.xticks(
        EXPECTED_BLOCK_SIZES
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
        / "blocksize_all_metrics.png"
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
    print("BLOCK SIZE ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()