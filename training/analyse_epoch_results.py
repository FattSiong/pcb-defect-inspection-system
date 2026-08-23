from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# SETTINGS
# =========================================================

RESULTS_CSV = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\epoch_testing\reduced_dataset_100_epochs"
    r"\results.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\epoch_testing\epoch_analysis_100"
)

EPOCH_LIMITS = [
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 100
]


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"results.csv not found:\n{RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    print("=" * 80)
    print("AVAILABLE COLUMNS")
    print("=" * 80)

    for column in df.columns:
        print(column)

    return df


# =========================================================
# PREPARE EPOCH NUMBERING
# =========================================================

def prepare_epochs(df):

    df = df.copy()

    # Some YOLO versions store first epoch as 0
    if df["epoch"].min() == 0:
        df["display_epoch"] = df["epoch"] + 1
    else:
        df["display_epoch"] = df["epoch"]

    return df


# =========================================================
# METRIC COLUMNS
# =========================================================

def get_metric_columns():

    return {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)"
    }


# =========================================================
# FIND BEST RESULT UP TO EACH EPOCH LIMIT
# =========================================================

def analyse_epoch_limits(df, columns):

    results = []

    for limit in EPOCH_LIMITS:

        # All epochs from 1 until current limit
        subset = df[
            df["display_epoch"] <= limit
        ]

        if subset.empty:
            continue

        # Find row with highest mAP50-95
        best_index = subset[
            columns["map50_95"]
        ].idxmax()

        best_row = subset.loc[best_index]

        results.append({
            "Training Limit": limit,
            "Best Epoch": int(
                best_row["display_epoch"]
            ),
            "Precision": best_row[
                columns["precision"]
            ],
            "Recall": best_row[
                columns["recall"]
            ],
            "mAP50": best_row[
                columns["map50"]
            ],
            "mAP50-95": best_row[
                columns["map50_95"]
            ]
        })

    comparison = pd.DataFrame(results)

    # Improvement compared with previous training limit
    comparison["mAP50-95 Improvement"] = (
        comparison["mAP50-95"].diff()
    )

    return comparison


# =========================================================
# EXACT CHECKPOINT RESULTS
# =========================================================

def analyse_exact_epochs(df, columns):

    rows = []

    for epoch in EPOCH_LIMITS:

        selected = df[
            df["display_epoch"] == epoch
        ]

        if selected.empty:
            continue

        row = selected.iloc[0]

        rows.append({
            "Epoch": epoch,
            "Precision": row[
                columns["precision"]
            ],
            "Recall": row[
                columns["recall"]
            ],
            "mAP50": row[
                columns["map50"]
            ],
            "mAP50-95": row[
                columns["map50_95"]
            ]
        })

    return pd.DataFrame(rows)


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(comparison):

    print("\n" + "=" * 100)
    print("BEST RESULT WITHIN EACH TRAINING LIMIT")
    print("=" * 100)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# =========================================================
# CREATE MAIN COMPARISON CHART
# =========================================================

def create_best_result_chart(comparison):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(11, 7))

    plt.plot(
        comparison["Training Limit"],
        comparison["mAP50-95"],
        marker="o",
        linewidth=2
    )

    # Add values beside points
    for _, row in comparison.iterrows():

        plt.text(
            row["Training Limit"],
            row["mAP50-95"] + 0.005,
            f'{row["mAP50-95"]:.4f}',
            ha="center"
        )

    plt.title(
        "Best Validation mAP50-95 by Training Epoch Limit"
    )

    plt.xlabel(
        "Maximum Training Epoch"
    )

    plt.ylabel(
        "Best Validation mAP50-95"
    )

    plt.xticks(
        EPOCH_LIMITS
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "best_map50_95_by_epoch_limit.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nChart saved to:\n{output_path}"
    )


# =========================================================
# CREATE ALL-METRICS CHART
# =========================================================

def create_metrics_chart(comparison):

    plt.figure(figsize=(11, 7))

    plt.plot(
        comparison["Training Limit"],
        comparison["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        comparison["Training Limit"],
        comparison["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        comparison["Training Limit"],
        comparison["mAP50"],
        marker="o",
        label="mAP50"
    )

    plt.plot(
        comparison["Training Limit"],
        comparison["mAP50-95"],
        marker="o",
        label="mAP50-95",
        linewidth=2
    )

    plt.title(
        "Best Validation Performance by Training Epoch Limit"
    )

    plt.xlabel(
        "Maximum Training Epoch"
    )

    plt.ylabel(
        "Performance Score"
    )

    plt.xticks(
        EPOCH_LIMITS
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
        "best_metrics_by_epoch_limit.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nChart saved to:\n{output_path}"
    )


# =========================================================
# SAVE CSV FILES
# =========================================================

def save_results(comparison, exact_results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "best_result_by_epoch_limit.csv",
        index=False
    )

    exact_results.to_csv(
        OUTPUT_DIR /
        "exact_epoch_results.csv",
        index=False
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("YOLOv8n 100-EPOCH ANALYSIS")
    print("=" * 80)

    df = load_results()

    df = prepare_epochs(df)

    columns = get_metric_columns()

    # Check columns exist
    for column in columns.values():

        if column not in df.columns:

            raise KeyError(
                f"Column not found: {column}"
            )

    # ---------------------------------------------
    # Best result within 10, 20, 30 ... 100 epochs
    # ---------------------------------------------

    comparison = analyse_epoch_limits(
        df,
        columns
    )

    print_results(
        comparison
    )

    # ---------------------------------------------
    # Exact results at Epoch 10, 20, 30 ...
    # ---------------------------------------------

    exact_results = analyse_exact_epochs(
        df,
        columns
    )

    print("\n" + "=" * 100)
    print("RESULT AT EXACT EPOCHS")
    print("=" * 100)

    print(
        exact_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Charts
    create_best_result_chart(
        comparison
    )

    create_metrics_chart(
        comparison
    )

    # Save results
    save_results(
        comparison,
        exact_results
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()