from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# SETTINGS
# =========================================================

RESULTS_CSV = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_blocksize"
    r"\blocksize_comparison.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_blocksize"
    r"\analysis"
)

# Only compare these five block sizes
BLOCK_SIZES = [11, 21, 31, 41, 51]


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find comparison CSV:\n{RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Make sure block size is integer
    df["Block Size"] = pd.to_numeric(
        df["Block Size"]
    ).astype(int)

    # Keep ONLY 11, 21, 31, 41, 51
    df = df[
        df["Block Size"].isin(BLOCK_SIZES)
    ].copy()

    # Sort correctly
    df = df.sort_values(
        "Block Size"
    ).reset_index(drop=True)

    return df


# =========================================================
# CHECK RESULTS
# =========================================================

def check_results(df):

    required_columns = [
        "Block Size",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise KeyError(
                f"Missing column: {column}"
            )

    found_blocks = set(
        df["Block Size"].tolist()
    )

    missing_blocks = [
        block
        for block in BLOCK_SIZES
        if block not in found_blocks
    ]

    if missing_blocks:

        raise ValueError(
            f"Missing results for block size(s): "
            f"{missing_blocks}\n\n"
            f"Current CSV contains: "
            f"{sorted(found_blocks)}"
        )


# =========================================================
# CALCULATE COMPARISON
# =========================================================

def prepare_comparison(df):

    df = df.copy()

    # Difference compared with previous block size
    df["mAP50-95 Change"] = (
        df["mAP50-95"].diff()
    )

    # Rank:
    # 1 = best mAP50-95
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
# PRINT COMPARISON TABLE
# =========================================================

def print_comparison(df):

    print("\n" + "=" * 110)
    print("STUDENT 1 - BLOCK SIZE VALIDATION COMPARISON")
    print("=" * 110)

    display = df[
        [
            "Block Size",
            "Precision",
            "Recall",
            "F1-score",
            "mAP50",
            "mAP50-95",
            "mAP50-95 Change",
            "Rank"
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# =========================================================
# FIND BEST BLOCK SIZE
# =========================================================

def find_best(df):

    best_index = (
        df["mAP50-95"].idxmax()
    )

    best = df.loc[best_index]

    print("\n" + "=" * 80)
    print("BEST BLOCK SIZE")
    print("=" * 80)

    print(
        f"Block Size : "
        f"{int(best['Block Size'])}"
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
        f"mAP50      : "
        f"{best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95   : "
        f"{best['mAP50-95']:.4f}"
    )

    if "Best Model" in df.columns:

        print("\nBest model:")
        print(best["Best Model"])

    return best


# =========================================================
# SAVE COMPARISON CSV
# =========================================================

def save_comparison(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        "blocksize_11_21_31_41_51_comparison.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nComparison CSV saved to:\n"
        f"{output_path}"
    )


# =========================================================
# CHART 1:
# mAP50-95 LINE CHART
# =========================================================

def create_map5095_chart(df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["Block Size"],
        df["mAP50-95"],
        marker="o",
        linewidth=2
    )

    # Add value labels
    for _, row in df.iterrows():

        plt.text(
            row["Block Size"],
            row["mAP50-95"] + 0.002,
            f"{row['mAP50-95']:.4f}",
            ha="center",
            fontsize=9
        )

    plt.title(
        "Validation mAP50-95 for Adaptive Gaussian Block Sizes"
    )

    plt.xlabel(
        "Block Size"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.xticks(
        BLOCK_SIZES
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "blocksize_map50_95_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nmAP50-95 chart saved to:\n"
        f"{output_path}"
    )


# =========================================================
# CHART 2:
# ALL PERFORMANCE METRICS
# =========================================================

def create_all_metrics_chart(df):

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        df["Block Size"],
        df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        df["Block Size"],
        df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        df["Block Size"],
        df["F1-score"],
        marker="o",
        label="F1-score"
    )

    plt.plot(
        df["Block Size"],
        df["mAP50"],
        marker="o",
        label="mAP50"
    )

    plt.plot(
        df["Block Size"],
        df["mAP50-95"],
        marker="o",
        linewidth=2,
        label="mAP50-95"
    )

    plt.title(
        "Validation Performance Across Block Sizes"
    )

    plt.xlabel(
        "Block Size"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.xticks(
        BLOCK_SIZES
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "blocksize_all_metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nAll-metrics chart saved to:\n"
        f"{output_path}"
    )


# =========================================================
# CHART 3:
# mAP50-95 BAR CHART
# =========================================================

def create_bar_chart(df):

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        df["Block Size"].astype(str),
        df["mAP50-95"]
    )

    for bar, value in zip(
        bars,
        df["mAP50-95"]
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 0.002,

            f"{value:.4f}",

            ha="center"
        )

    plt.title(
        "mAP50-95 Comparison of Block Size Configurations"
    )

    plt.xlabel(
        "Block Size"
    )

    plt.ylabel(
        "Validation mAP50-95"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "blocksize_map50_95_bar_chart.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nBar chart saved to:\n"
        f"{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_results()

    check_results(
        df
    )

    df = prepare_comparison(
        df
    )

    print_comparison(
        df
    )

    find_best(
        df
    )

    save_comparison(
        df
    )

    create_map5095_chart(
        df
    )

    create_all_metrics_chart(
        df
    )

    create_bar_chart(
        df
    )

    print("\n" + "=" * 80)
    print("BLOCK SIZE COMPARISON COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()