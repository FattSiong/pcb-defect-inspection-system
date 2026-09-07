from pathlib import Path

from build_bilateral_morphology_dataset import (
    create_data_yaml,
    process_split,
    save_processing_config,
    verify_split,
)


# =========================================================
# SETTINGS — CHANGE PARAMETERS HERE
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\hangg\Downloads\pcb_dataset_reduced"
)

OUTPUT_PARENT = Path(
    r"C:\Users\hangg\Downloads"
)

# Change this name for every parameter configuration.
CONFIGURATION_NAME = "stage3_w3"

# Do not include test during parameter tuning.
SPLITS = (
    "train",
    "val",
)

# ---------------------------------------------------------
# Bilateral-filter parameters
# ---------------------------------------------------------

BILATERAL_D = 3
SIGMA_COLOR = 15
SIGMA_SPACE = 3

# ---------------------------------------------------------
# Multi-scale morphology parameters
# ---------------------------------------------------------

KERNEL_SIZES = (
    7,
    15,
    31,
)

ALPHA = 1.0
BETA = 1.0

# Set to True only if replacing an existing configuration.
ALLOW_OVERWRITE = False


# =========================================================
# GENERATED OUTPUT PATH
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
            f"Source dataset not found:\n"
            f"{SOURCE_DATASET}"
        )

    if SOURCE_DATASET.resolve() == OUTPUT_DATASET.resolve():
        raise ValueError(
            "Source and output datasets cannot be the same."
        )

    if BILATERAL_D <= 0:
        raise ValueError(
            "BILATERAL_D must be greater than zero."
        )

    if SIGMA_COLOR <= 0:
        raise ValueError(
            "SIGMA_COLOR must be greater than zero."
        )

    if SIGMA_SPACE <= 0:
        raise ValueError(
            "SIGMA_SPACE must be greater than zero."
        )

    for size in KERNEL_SIZES:
        if size <= 1 or size % 2 == 0:
            raise ValueError(
                "Every morphology kernel size must be "
                "an odd integer greater than one."
            )

    if ALPHA < 0 or BETA < 0:
        raise ValueError(
            "ALPHA and BETA cannot be negative."
        )

    if (
        OUTPUT_DATASET.exists()
        and any(OUTPUT_DATASET.iterdir())
        and not ALLOW_OVERWRITE
    ):
        raise FileExistsError(
            f"Output directory is not empty:\n"
            f"{OUTPUT_DATASET}\n\n"
            f"Change CONFIGURATION_NAME or set "
            f"ALLOW_OVERWRITE = True."
        )


# =========================================================
# PRINT CURRENT CONFIGURATION
# =========================================================

def print_configuration():
    print("=" * 70)
    print("GENERATING PARAMETER-TUNING DATASET")
    print("=" * 70)

    print(f"Configuration : {CONFIGURATION_NAME}")
    print(f"Source        : {SOURCE_DATASET}")
    print(f"Output        : {OUTPUT_DATASET}")
    print(f"Splits        : {SPLITS}")

    print()
    print("Bilateral filtering")
    print(f"  Diameter    : {BILATERAL_D}")
    print(f"  Sigma colour: {SIGMA_COLOR}")
    print(f"  Sigma space : {SIGMA_SPACE}")

    print()
    print("Multi-scale morphology")
    print(f"  Kernels     : {KERNEL_SIZES}")
    print(f"  Alpha       : {ALPHA}")
    print(f"  Beta        : {BETA}")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():
    validate_settings()
    print_configuration()

    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = []

    for split in SPLITS:
        result = process_split(
            source_root=SOURCE_DATASET,
            output_root=OUTPUT_DATASET,
            split=split,
            bilateral_d=BILATERAL_D,
            sigma_color=SIGMA_COLOR,
            sigma_space=SIGMA_SPACE,
            kernel_sizes=KERNEL_SIZES,
            alpha=ALPHA,
            beta=BETA,
        )

        statistics.append(result)

    print()
    print("=" * 70)
    print("VERIFYING DATASET")
    print("=" * 70)

    for split in SPLITS:
        verify_split(
            OUTPUT_DATASET,
            split,
        )

    create_data_yaml(
        OUTPUT_DATASET,
        SPLITS,
    )

    save_processing_config(
        source_root=SOURCE_DATASET,
        output_root=OUTPUT_DATASET,
        splits=SPLITS,
        bilateral_d=BILATERAL_D,
        sigma_color=SIGMA_COLOR,
        sigma_space=SIGMA_SPACE,
        kernel_sizes=KERNEL_SIZES,
        alpha=ALPHA,
        beta=BETA,
        statistics=statistics,
    )

    print()
    print("=" * 70)
    print("PARAMETER DATASET COMPLETED")
    print("=" * 70)
    print(f"Dataset: {OUTPUT_DATASET}")
    print(
        f"YAML   : "
        f"{OUTPUT_DATASET / 'data.yaml'}"
    )


if __name__ == "__main__":
    main()