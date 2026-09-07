from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
)

DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

RESULTS_DIR = PROJECT_ROOT / "results"

OVERALL_METRICS_CSV = (
    RESULTS_DIR
    / "student1_overall_test_metrics.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "processing_time"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TECHNIQUE A FINAL PARAMETERS
# ============================================================

CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID = (8, 8)

ADAPTIVE_BLOCK_SIZE = 41
ADAPTIVE_C = 2


# ============================================================
# TIMING SETTINGS
# ============================================================

# Run the complete test set multiple times to reduce timing noise.
REPEATS = 3

# Warm-up images are processed before timing begins.
WARMUP_IMAGES = 20


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# FIND TEST IMAGE DIRECTORY
# ============================================================

def find_test_images_directory():

    possible_directories = [
        DATASET_ROOT / "test" / "images",
        DATASET_ROOT / "test",
    ]

    for directory in possible_directories:

        if not directory.exists():
            continue

        image_files = [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

        if image_files:
            return directory

    raise FileNotFoundError(
        "Could not find the test images. "
        f"Checked under: {DATASET_ROOT}"
    )


# ============================================================
# GET TEST IMAGES
# ============================================================

def get_test_images(
    test_directory
):

    image_paths = sorted(
        [
            path
            for path in test_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if not image_paths:

        raise FileNotFoundError(
            "No test images were found."
        )

    return image_paths


# ============================================================
# TECHNIQUE A PREPROCESSING
# ============================================================

def apply_technique_a(
    image
):

    # --------------------------------------------------------
    # Step 1: Convert original PCB image to grayscale
    # --------------------------------------------------------

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Step 2: CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    clahe_image = clahe.apply(
        grayscale
    )


    # --------------------------------------------------------
    # Step 3: Adaptive Gaussian Thresholding
    # --------------------------------------------------------

    thresholded_image = cv2.adaptiveThreshold(
        clahe_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    return thresholded_image


# ============================================================
# WARM-UP
# ============================================================

def warm_up(
    image_paths
):

    number_of_warmup_images = min(
        WARMUP_IMAGES,
        len(image_paths)
    )

    print(
        f"\nWarming up with "
        f"{number_of_warmup_images} images..."
    )

    for image_path in (
        image_paths[
            :number_of_warmup_images
        ]
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        _ = apply_technique_a(
            image
        )


# ============================================================
# MEASURE PREPROCESSING TIME
# ============================================================

def measure_preprocessing_time(
    image_paths
):

    timing_records = []

    total_processed = 0

    print()
    print("=" * 80)
    print(
        "MEASURING TECHNIQUE A "
        "PREPROCESSING TIME"
    )
    print("=" * 80)

    for repeat in range(
        1,
        REPEATS + 1
    ):

        print(
            f"\nRepeat "
            f"{repeat}/{REPEATS}"
        )

        for image_number, image_path in enumerate(
            image_paths,
            start=1
        ):

            # ------------------------------------------------
            # Image loading is deliberately NOT included
            # in preprocessing time.
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                print(
                    f"Warning: Could not read "
                    f"{image_path}"
                )

                continue


            # ------------------------------------------------
            # Start timing immediately before Technique A
            # ------------------------------------------------

            start_time = (
                time.perf_counter_ns()
            )

            _ = apply_technique_a(
                image
            )

            end_time = (
                time.perf_counter_ns()
            )


            # ------------------------------------------------
            # Convert nanoseconds to milliseconds
            # ------------------------------------------------

            preprocessing_time_ms = (
                end_time
                - start_time
            ) / 1_000_000


            timing_records.append(
                {
                    "Repeat":
                        repeat,

                    "Image":
                        image_path.name,

                    "Preprocessing Time (ms)":
                        preprocessing_time_ms,
                }
            )

            total_processed += 1


            # ------------------------------------------------
            # Progress display
            # ------------------------------------------------

            if (
                image_number % 100 == 0
                or
                image_number
                == len(image_paths)
            ):

                print(
                    f"  Processed "
                    f"{image_number}/"
                    f"{len(image_paths)} "
                    f"images"
                )


    timing_df = pd.DataFrame(
        timing_records
    )

    return timing_df


# ============================================================
# READ YOLO TIMING FROM EXISTING FINAL TEST CSV
# ============================================================

def read_yolo_metrics():

    if not OVERALL_METRICS_CSV.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{OVERALL_METRICS_CSV}"
        )

    metrics_df = pd.read_csv(
        OVERALL_METRICS_CSV
    )

    if metrics_df.empty:

        raise ValueError(
            "The overall test metrics CSV "
            "is empty."
        )

    row = metrics_df.iloc[0]

    required_columns = [
        "YOLO Preprocess Time (ms/image)",
        "Inference Time (ms/image)",
        "Postprocess Time (ms/image)",
        "Total YOLO Time (ms/image)",
        "mAP50-95",
    ]

    for column in required_columns:

        if column not in metrics_df.columns:

            raise ValueError(
                f"Missing column in "
                f"{OVERALL_METRICS_CSV.name}: "
                f"{column}"
            )

    return {
        "YOLO Preprocess Time (ms/image)":
            float(
                row[
                    "YOLO Preprocess Time (ms/image)"
                ]
            ),

        "Inference Time (ms/image)":
            float(
                row[
                    "Inference Time (ms/image)"
                ]
            ),

        "Postprocess Time (ms/image)":
            float(
                row[
                    "Postprocess Time (ms/image)"
                ]
            ),

        "Total YOLO Time (ms/image)":
            float(
                row[
                    "Total YOLO Time (ms/image)"
                ]
            ),

        "mAP50-95":
            float(
                row[
                    "mAP50-95"
                ]
            ),
    }


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    timing_df,
    number_of_test_images,
    yolo_metrics
):

    preprocessing_times = (
        timing_df[
            "Preprocessing Time (ms)"
        ]
    )

    average_preprocessing = (
        preprocessing_times.mean()
    )

    median_preprocessing = (
        preprocessing_times.median()
    )

    std_preprocessing = (
        preprocessing_times.std()
    )

    minimum_preprocessing = (
        preprocessing_times.min()
    )

    maximum_preprocessing = (
        preprocessing_times.max()
    )


    # ========================================================
    # IMPORTANT
    #
    # Total pipeline time is estimated as:
    #
    # Custom Technique A preprocessing
    # +
    # YOLO preprocessing
    # +
    # YOLO inference
    # +
    # YOLO postprocessing
    #
    # The custom CLAHE + Adaptive Gaussian Thresholding
    # stage was previously performed offline, so it is not
    # included in Ultralytics' reported YOLO preprocessing
    # time.
    # ========================================================

    total_pipeline_time = (
        average_preprocessing
        +
        yolo_metrics[
            "Total YOLO Time (ms/image)"
        ]
    )


    estimated_fps = (
        1000
        / total_pipeline_time
    )


    summary_df = pd.DataFrame(
        [
            {
                "Student":
                    "Student 1",

                "Technique":
                    (
                        "CLAHE + Adaptive "
                        "Gaussian Thresholding"
                    ),

                "Number of Test Images":
                    number_of_test_images,

                "Timing Repeats":
                    REPEATS,

                "CLAHE Clip Limit":
                    CLAHE_CLIP_LIMIT,

                "CLAHE Tile Grid":
                    (
                        f"{CLAHE_TILE_GRID[0]}"
                        f"x"
                        f"{CLAHE_TILE_GRID[1]}"
                    ),

                "Adaptive Block Size":
                    ADAPTIVE_BLOCK_SIZE,

                "Adaptive C":
                    ADAPTIVE_C,

                "Average Custom Preprocessing "
                "Time (ms/image)":
                    average_preprocessing,

                "Median Custom Preprocessing "
                "Time (ms/image)":
                    median_preprocessing,

                "Std Custom Preprocessing "
                "Time (ms/image)":
                    std_preprocessing,

                "Minimum Custom Preprocessing "
                "Time (ms/image)":
                    minimum_preprocessing,

                "Maximum Custom Preprocessing "
                "Time (ms/image)":
                    maximum_preprocessing,

                "YOLO Preprocess Time "
                "(ms/image)":
                    yolo_metrics[
                        "YOLO Preprocess Time "
                        "(ms/image)"
                    ],

                "YOLO Inference Time "
                "(ms/image)":
                    yolo_metrics[
                        "Inference Time "
                        "(ms/image)"
                    ],

                "YOLO Postprocess Time "
                "(ms/image)":
                    yolo_metrics[
                        "Postprocess Time "
                        "(ms/image)"
                    ],

                "Total YOLO Time "
                "(ms/image)":
                    yolo_metrics[
                        "Total YOLO Time "
                        "(ms/image)"
                    ],

                "Estimated Total Pipeline "
                "Time (ms/image)":
                    total_pipeline_time,

                "Estimated Pipeline FPS":
                    estimated_fps,

                "Final Test mAP50-95":
                    yolo_metrics[
                        "mAP50-95"
                    ],
            }
        ]
    )

    return summary_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    timing_df,
    summary_df
):

    per_image_path = (
        OUTPUT_DIR
        / "student1_preprocessing_per_image.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "student1_processing_time_summary.csv"
    )

    timing_df.to_csv(
        per_image_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    return (
        per_image_path,
        summary_path
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary_df
):

    row = summary_df.iloc[0]

    print()
    print("=" * 80)
    print(
        "TECHNIQUE A COMPUTATIONAL "
        "EFFICIENCY SUMMARY"
    )
    print("=" * 80)

    print(
        f"Test images: "
        f"{int(row['Number of Test Images'])}"
    )

    print(
        f"Timing repeats: "
        f"{int(row['Timing Repeats'])}"
    )

    print()
    print(
        "Custom Technique A preprocessing:"
    )

    print(
        f"  Average: "
        f"{row['Average Custom Preprocessing Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Median:  "
        f"{row['Median Custom Preprocessing Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Std Dev: "
        f"{row['Std Custom Preprocessing Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print()
    print(
        "Existing YOLO timing:"
    )

    print(
        f"  YOLO preprocess: "
        f"{row['YOLO Preprocess Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Inference:       "
        f"{row['YOLO Inference Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Postprocess:     "
        f"{row['YOLO Postprocess Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Total YOLO:      "
        f"{row['Total YOLO Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print()
    print(
        "Estimated complete pipeline:"
    )

    print(
        f"  Total pipeline: "
        f"{row['Estimated Total Pipeline Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Estimated FPS:  "
        f"{row['Estimated Pipeline FPS']:.2f}"
    )

    print()
    print(
        f"Final mAP50-95: "
        f"{row['Final Test mAP50-95']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "STUDENT 1 - TECHNIQUE A "
        "PROCESSING TIME ANALYSIS"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Locate test data
    # --------------------------------------------------------

    test_directory = (
        find_test_images_directory()
    )

    image_paths = (
        get_test_images(
            test_directory
        )
    )

    print(
        f"\nTest image directory:\n"
        f"{test_directory}"
    )

    print(
        f"\nNumber of test images found: "
        f"{len(image_paths)}"
    )


    # --------------------------------------------------------
    # Warm up OpenCV
    # --------------------------------------------------------

    warm_up(
        image_paths
    )


    # --------------------------------------------------------
    # Measure actual Technique A preprocessing
    # --------------------------------------------------------

    timing_df = (
        measure_preprocessing_time(
            image_paths
        )
    )


    # --------------------------------------------------------
    # Read existing YOLO timing
    # --------------------------------------------------------

    yolo_metrics = (
        read_yolo_metrics()
    )


    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary_df = (
        create_summary(
            timing_df,
            len(image_paths),
            yolo_metrics
        )
    )


    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    (
        per_image_path,
        summary_path
    ) = save_results(
        timing_df,
        summary_df
    )


    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print_summary(
        summary_df
    )


    print()
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print(
        f"\nPer-image timing:\n"
        f"{per_image_path}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_path}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()