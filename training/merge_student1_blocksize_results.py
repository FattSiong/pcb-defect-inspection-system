from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

RESULTS_CSV = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_blocksize"
    r"\blocksize_comparison.csv"
)

OUTPUT_CSV = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\student1_blocksize"
    r"\blocksize_comparison_all.csv"
)


# =========================================================
# EXISTING RESULTS FOR BLOCK SIZE 11, 21, 31
# =========================================================

old_results = pd.DataFrame([
    {
        "Block Size": 11,
        "Precision": 0.9320,
        "Recall": 0.8410,
        "F1-score": 0.8841,
        "mAP50": 0.9092,
        "mAP50-95": 0.4631,
    },
    {
        "Block Size": 21,
        "Precision": 0.9363,
        "Recall": 0.9010,
        "F1-score": 0.9183,
        "mAP50": 0.9338,
        "mAP50-95": 0.4833,
    },
    {
        "Block Size": 31,
        "Precision": 0.9451,
        "Recall": 0.9175,
        "F1-score": 0.9311,
        "mAP50": 0.9466,
        "mAP50-95": 0.4882,
    },
])


# =========================================================
# LOAD NEW 41 / 51 RESULTS
# =========================================================

def load_new_results():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    df.columns = df.columns.str.strip()

    df["Block Size"] = pd.to_numeric(
        df["Block Size"]
    ).astype(int)

    # Only take 41 and 51
    df = df[
        df["Block Size"].isin([41, 51])
    ].copy()

    if len(df) != 2:
        print(
            "WARNING: Expected results for both "
            "block size 41 and 51."
        )

    return df


# =========================================================
# MERGE RESULTS
# =========================================================

def merge_results():

    new_results = load_new_results()

    # Keep only columns needed for the comparison
    comparison_columns = [
        "Block Size",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
    ]

    new_results = new_results[
        comparison_columns
    ]

    combined = pd.concat(
        [
            old_results,
            new_results
        ],
        ignore_index=True
    )

    combined = combined.sort_values(
        "Block Size"
    ).reset_index(drop=True)

    return combined


# =========================================================
# PRINT AND SAVE
# =========================================================

def main():

    combined = merge_results()

    print("\n" + "=" * 95)
    print("COMPLETE BLOCK SIZE VALIDATION COMPARISON")
    print("=" * 95)

    print(
        combined.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Find best block size
    best_index = (
        combined["mAP50-95"].idxmax()
    )

    best = combined.loc[
        best_index
    ]

    print("\n" + "=" * 95)
    print("BEST BLOCK SIZE")
    print("=" * 95)

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

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nComplete comparison saved to:\n"
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()