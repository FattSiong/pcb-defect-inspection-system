from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL mAP50-95 RESULTS
# ============================================================

approaches = [
    "Original Baseline",
    "Technique A",
    "Technique B",
    "Technique C",
    "Technique D",
]

map50_95_values = [
    0.5038,  # Original image baseline
    0.4783,  # Technique A
    0.3185,  # Technique B
    0.4910,  # Technique C
    0.5011,  # Technique D
]


# ============================================================
# CREATE GRAPH
# ============================================================

figure, axis = plt.subplots(
    figsize=(11, 7)
)

bars = axis.bar(
    approaches,
    map50_95_values
)


# ============================================================
# VALUE LABELS
# ============================================================

for bar, value in zip(
    bars,
    map50_95_values
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


# ============================================================
# GRAPH FORMATTING
# ============================================================

axis.set_title(
    "Final Test mAP50-95 Comparison Including "
    "Original Image Baseline",
    fontsize=15
)

axis.set_xlabel(
    "Approach",
    fontsize=12
)

axis.set_ylabel(
    "mAP50-95",
    fontsize=12
)

axis.set_ylim(
    0,
    0.60
)

axis.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

axis.tick_params(
    axis="x",
    labelsize=10
)

axis.tick_params(
    axis="y",
    labelsize=10
)

figure.tight_layout()


# ============================================================
# SAVE GRAPH
# ============================================================

output_path = (
    OUTPUT_DIR
    / "baseline_vs_four_techniques_map50_95.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 75)
print("FINAL TEST mAP50-95 COMPARISON")
print("=" * 75)

for approach, value in zip(
    approaches,
    map50_95_values
):
    print(
        f"{approach:<20} "
        f"{value:.4f}"
    )

print()
print(
    "Best overall approach: "
    "Original Image Baseline "
    "(mAP50-95 = 0.5038)"
)

print(
    "Best preprocessing technique: "
    "Technique D "
    "(mAP50-95 = 0.5011)"
)

print(
    "Difference between baseline and Technique D: "
    f"{0.5038 - 0.5011:.4f}"
)

print()
print(
    f"Graph saved to:\n{output_path}"
)