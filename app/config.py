from pathlib import Path


# ============================================================
# APPLICATION PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    APP_DIR
    / "models"
    / "technique_d_best.pt"
)


# ============================================================
# YOLO CONFIGURATION
# ============================================================

MODEL_IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# TECHNIQUE D FINAL CONFIGURATION
# ============================================================

# Bilateral Filtering
BILATERAL_DIAMETER = 3
BILATERAL_SIGMA_COLOR = 15
BILATERAL_SIGMA_SPACE = 3

# Multi-scale morphology
MORPH_KERNEL_SIZES = [
    7,
    15,
    31,
]

# Enhancement weights
MORPH_ALPHA = 0.7
MORPH_BETA = 0.7


# ============================================================
# PCB DEFECT CLASSES
# ============================================================

DEFECT_CLASSES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}