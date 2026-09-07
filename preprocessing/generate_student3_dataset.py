"""Generate one editable Gaussian-LBP parameter dataset.

Change only the values in the SETTINGS section, then run this file.
The source images are never modified.
"""

from pathlib import Path

from build_gaussian_lbp_dataset import (
    create_data_yaml,
    process_split,
    save_processing_config,
    verify_split,
)


# =========================================================
# SETTINGS - CHANGE PARAMETERS HERE
# =========================================================

# Root of the original YOLO dataset. It must contain folders such as:
# train/images, train/labels, val/images and val/labels.
SOURCE_DATASET = Path(
    r"C:\Users\ASUS\Downloads\pcb_dataset_reduced"
)

# Every generated dataset will be placed inside this directory.
OUTPUT_PARENT = Path(
    r"C:\Users\ASUS\Downloads"
)

# Use a different name for every parameter configuration.
# Example names: gaussian_lbp_s1, gaussian_lbp_s2, gaussian_lbp_w015
CONFIGURATION_NAME = "gaussian_lbp_final_w1"

# Use only train and val while tuning parameters.
# Add "test" only after the final configuration has been selected.
SPLITS = (
    "train",
    "val",
    "test",
)

# ---------------------------------------------------------
# Gaussian-filter parameters
# ---------------------------------------------------------

# Must be an odd integer greater than one: 3, 5, 7, ...
GAUSSIAN_KERNEL = 3

# Must be greater than zero.
GAUSSIAN_SIGMA = 0.8

# ---------------------------------------------------------
# Single-scale uniform-LBP parameters
# ---------------------------------------------------------

# Recommended single-scale starting configuration: P=8, R=1.
LBP_POINTS = 8
LBP_RADIUS = 1.0

# Amount of normalized LBP texture fused into LAB lightness.
# Must be between 0.0 and 1.0. The Gaussian contribution is
# automatically calculated as 1.0 - LBP_WEIGHT.
LBP_WEIGHT = 0.15

# Set to True only when intentionally replacing matching files in an
# existing generated dataset. Files are not deleted automatically.
ALLOW_OVERWRITE = False


# =========================================================
# GENERATED OUTPUT PATH - DO NOT EDIT
# =========================================================

OUTPUT_DATASET = (
    OUTPUT_PARENT /
    f"pcb_dataset_{CONFIGURATION_NAME}"
)


# =========================================================
# VALIDATE SETTINGS
# =========================================================

def validate_settings():
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE_DATASET}"
        )

    if SOURCE_DATASET.resolve() == OUTPUT_DATASET.resolve():
        raise ValueError(
            "Source and output datasets cannot be the same."
        )

    if not CONFIGURATION_NAME.strip():
        raise ValueError("CONFIGURATION_NAME cannot be empty.")

    if any(separator in CONFIGURATION_NAME for separator in ("/", "\\")):
        raise ValueError(
            "CONFIGURATION_NAME must not contain directory separators."
        )

    if not SPLITS:
        raise ValueError("At least one dataset split is required.")

    valid_splits = {"train", "val", "test"}
    invalid_splits = set(SPLITS) - valid_splits

    if invalid_splits:
        raise ValueError(
            f"Unsupported splits: {sorted(invalid_splits)}"
        )

    if len(SPLITS) != len(set(SPLITS)):
        raise ValueError("SPLITS cannot contain duplicate values.")

    if (
        not isinstance(GAUSSIAN_KERNEL, int)
        or GAUSSIAN_KERNEL <= 1
        or GAUSSIAN_KERNEL % 2 == 0
    ):
        raise ValueError(
            "GAUSSIAN_KERNEL must be an odd integer greater than one."
        )

    if GAUSSIAN_SIGMA <= 0:
        raise ValueError("GAUSSIAN_SIGMA must be greater than zero.")

    if (
        not isinstance(LBP_POINTS, int)
        or LBP_POINTS < 4
        or LBP_POINTS > 253
    ):
        raise ValueError(
            "LBP_POINTS must be an integer between 4 and 253."
        )

    if LBP_RADIUS <= 0:
        raise ValueError("LBP_RADIUS must be greater than zero.")

    if not 0.0 <= LBP_WEIGHT <= 1.0:
        raise ValueError("LBP_WEIGHT must be between zero and one.")

    if (
        OUTPUT_DATASET.exists()
        and any(OUTPUT_DATASET.iterdir())
        and not ALLOW_OVERWRITE
    ):
        raise FileExistsError(
            f"Output directory is not empty:\n{OUTPUT_DATASET}\n\n"
            "Change CONFIGURATION_NAME or set "
            "ALLOW_OVERWRITE = True."
        )


# =========================================================
# PRINT CURRENT CONFIGURATION
# =========================================================

def print_configuration():
    print("=" * 72)
    print("GENERATING GAUSSIAN + SINGLE-SCALE LBP DATASET")
    print("=" * 72)
    print(f"Configuration   : {CONFIGURATION_NAME}")
    print(f"Source          : {SOURCE_DATASET}")
    print(f"Output          : {OUTPUT_DATASET}")
    print(f"Splits          : {SPLITS}")
    print()
    print("Gaussian filtering")
    print(f"  Kernel        : {GAUSSIAN_KERNEL}")
    print(f"  Sigma         : {GAUSSIAN_SIGMA}")
    print()
    print("Uniform LBP")
    print(f"  Points (P)    : {LBP_POINTS}")
    print(f"  Radius (R)    : {LBP_RADIUS}")
    print(f"  LBP weight    : {LBP_WEIGHT}")
    print(f"  Gaussian weight: {1.0 - LBP_WEIGHT}")
    print("=" * 72)


# =========================================================
# MAIN
# =========================================================

def main():
    validate_settings()
    print_configuration()
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)

    statistics = []

    for split in SPLITS:
        statistics.append(
            process_split(
                source_root=SOURCE_DATASET,
                output_root=OUTPUT_DATASET,
                split=split,
                gaussian_kernel=GAUSSIAN_KERNEL,
                gaussian_sigma=GAUSSIAN_SIGMA,
                lbp_points=LBP_POINTS,
                lbp_radius=LBP_RADIUS,
                lbp_weight=LBP_WEIGHT,
            )
        )

    print()
    print("=" * 72)
    print("VERIFYING GENERATED DATASET")
    print("=" * 72)

    for split in SPLITS:
        verify_split(OUTPUT_DATASET, split)

    create_data_yaml(OUTPUT_DATASET, SPLITS)
    save_processing_config(
        source_root=SOURCE_DATASET,
        output_root=OUTPUT_DATASET,
        splits=SPLITS,
        gaussian_kernel=GAUSSIAN_KERNEL,
        gaussian_sigma=GAUSSIAN_SIGMA,
        lbp_points=LBP_POINTS,
        lbp_radius=LBP_RADIUS,
        lbp_weight=LBP_WEIGHT,
        statistics=statistics,
    )

    print()
    print("=" * 72)
    print("PARAMETER DATASET COMPLETED")
    print("=" * 72)
    print(f"Dataset: {OUTPUT_DATASET}")
    print(f"YAML   : {OUTPUT_DATASET / 'data.yaml'}")


if __name__ == "__main__":
    main()

