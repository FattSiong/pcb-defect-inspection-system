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

# Final Technique B YOLO test metrics
OVERALL_METRICS_CSV = (
    PROJECT_ROOT
    / "runs"
    / "student2_final_test"
    / "technique_b_test"
    / "overall_test_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "processing_time"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TECHNIQUE B FINAL PARAMETERS
# ============================================================

# Canny Edge Detection
CANNY_THRESH = (50, 150)

# Probabilistic Hough Transform
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 5

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180


# ============================================================
# TIMING SETTINGS
# ============================================================

# Complete test set is processed three times
REPEATS = 3

# Warm-up avoids measuring initial OpenCV overhead
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
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]

        if image_files:
            return directory

    raise FileNotFoundError(
        "Could not find the test images.\n"
        f"Checked under: {DATASET_ROOT}"
    )


# ============================================================
# GET TEST IMAGE PATHS
# ============================================================

def get_test_images(test_directory):

    image_paths = sorted(
        [
            path
            for path in test_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )

    if not image_paths:
        raise FileNotFoundError(
            "No test images were found."
        )

    return image_paths


# ============================================================
# TECHNIQUE B PREPROCESSING
# ============================================================

def apply_technique_b(image):

    # --------------------------------------------------------
    # Step 1: Grayscale conversion
    # --------------------------------------------------------

    if image.ndim == 3:

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        grayscale = image.copy()


    # --------------------------------------------------------
    # Step 2: Gaussian Blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # Step 3: Canny Edge Detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        CANNY_THRESH[0],
        CANNY_THRESH[1]
    )


    # --------------------------------------------------------
    # Step 4: Probabilistic Hough Transform
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )


    # --------------------------------------------------------
    # Step 5: Convert edge image to BGR
    # --------------------------------------------------------

    output_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )


    # --------------------------------------------------------
    # Step 6: Draw detected Hough lines
    #
    # IMPORTANT:
    # HoughLinesP may return:
    #
    # (N, 1, 4)
    #
    # or
    #
    # (N, 4)
    #
    # depending on the OpenCV environment.
    #
    # reshape(-1, 4) makes both formats safe.
    # --------------------------------------------------------

    if lines is not None:

        lines = np.asarray(
            lines,
            dtype=np.int32
        ).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            cv2.line(
                output_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )


    return output_bgr


# ============================================================
# WARM-UP
# ============================================================

def warm_up(image_paths):

    number_of_warmup_images = min(
        WARMUP_IMAGES,
        len(image_paths)
    )

    print(
        f"\nWarming up with "
        f"{number_of_warmup_images} images..."
    )

    processed = 0

    for image_path in image_paths[:number_of_warmup_images]:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Warning: Could not read "
                f"{image_path}"
            )

            continue

        _ = apply_technique_b(
            image
        )

        processed += 1

    print(
        f"Warm-up completed with "
        f"{processed} images."
    )


# ============================================================
# MEASURE PREPROCESSING TIME
# ============================================================

def measure_preprocessing_time(image_paths):

    timing_records = []

    print()
    print("=" * 80)
    print(
        "MEASURING TECHNIQUE B "
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

        repeat_times = []

        for image_number, image_path in enumerate(
            image_paths,
            start=1
        ):

            # ------------------------------------------------
            # Image loading is NOT included in preprocessing
            # time.
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
            # Start timing immediately before Technique B
            # ------------------------------------------------

            start_time = (
                time.perf_counter_ns()
            )

            _ = apply_technique_b(
                image
            )

            end_time = (
                time.perf_counter_ns()
            )


            # ------------------------------------------------
            # Nanoseconds → milliseconds
            # ------------------------------------------------

            preprocessing_time_ms = (
                end_time - start_time
            ) / 1_000_000


            repeat_times.append(
                preprocessing_time_ms
            )


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


            # ------------------------------------------------
            # Progress display
            # ------------------------------------------------

            if (
                image_number % 100 == 0
                or image_number == len(image_paths)
            ):

                print(
                    f"  Processed "
                    f"{image_number}/"
                    f"{len(image_paths)} images"
                )


        if repeat_times:

            print(
                f"  Repeat {repeat} average: "
                f"{np.mean(repeat_times):.4f} "
                f"ms/image"
            )


    timing_df = pd.DataFrame(
        timing_records
    )

    if timing_df.empty:

        raise RuntimeError(
            "No preprocessing timing records "
            "were generated."
        )

    return timing_df


# ============================================================
# READ EXISTING YOLO TIMING
# ============================================================

def read_yolo_metrics():

    if not OVERALL_METRICS_CSV.exists():

        raise FileNotFoundError(
            "Could not find the Technique B "
            "overall test metrics CSV:\n"
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


    required_columns = [
        "YOLO Preprocess Time (ms/image)",
        "Inference Time (ms/image)",
        "Postprocess Time (ms/image)",
        "Total YOLO Time (ms/image)",
        "mAP50-95",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in metrics_df.columns
    ]


    if missing_columns:

        raise ValueError(
            "The following required columns "
            "are missing from "
            f"{OVERALL_METRICS_CSV.name}:\n"
            + "\n".join(missing_columns)
        )


    row = metrics_df.iloc[0]


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
    # Complete pipeline time
    #
    # Technique B custom preprocessing
    # +
    # YOLO preprocessing
    # +
    # YOLO inference
    # +
    # YOLO postprocessing
    #
    # Since Total YOLO Time already contains the three YOLO
    # stages, the complete estimate is:
    #
    # Custom preprocessing + Total YOLO Time
    # ========================================================

    total_pipeline_time = (
        average_preprocessing
        +
        yolo_metrics[
            "Total YOLO Time (ms/image)"
        ]
    )


    estimated_fps = (
        1000.0
        / total_pipeline_time
    )


    summary_df = pd.DataFrame(
        [
            {
                "Student":
                    "Student 2",

                "Technique":
                    (
                        "Canny Edge Detection + "
                        "Hough Transform"
                    ),

                "Number of Test Images":
                    number_of_test_images,

                "Timing Repeats":
                    REPEATS,

                "Canny Thresholds":
                    (
                        f"{CANNY_THRESH[0]}"
                        f"-"
                        f"{CANNY_THRESH[1]}"
                    ),

                "Hough Rho":
                    HOUGH_RHO,

                "Hough Theta (degrees)":
                    1,

                "Hough Threshold":
                    HOUGH_THRESHOLD,

                "Min Line Length":
                    HOUGH_MIN_LINE_LENGTH,

                "Max Line Gap":
                    HOUGH_MAX_LINE_GAP,

                "Average Custom Preprocessing Time (ms/image)":
                    average_preprocessing,

                "Median Custom Preprocessing Time (ms/image)":
                    median_preprocessing,

                "Std Custom Preprocessing Time (ms/image)":
                    std_preprocessing,

                "Minimum Custom Preprocessing Time (ms/image)":
                    minimum_preprocessing,

                "Maximum Custom Preprocessing Time (ms/image)":
                    maximum_preprocessing,

                "YOLO Preprocess Time (ms/image)":
                    yolo_metrics[
                        "YOLO Preprocess Time (ms/image)"
                    ],

                "YOLO Inference Time (ms/image)":
                    yolo_metrics[
                        "Inference Time (ms/image)"
                    ],

                "YOLO Postprocess Time (ms/image)":
                    yolo_metrics[
                        "Postprocess Time (ms/image)"
                    ],

                "Total YOLO Time (ms/image)":
                    yolo_metrics[
                        "Total YOLO Time (ms/image)"
                    ],

                "Estimated Total Pipeline Time (ms/image)":
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
        / "student2_preprocessing_per_image.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "student2_processing_time_summary.csv"
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
# PRINT REPEAT STATISTICS
# ============================================================

def print_repeat_statistics(
    timing_df
):

    print()
    print("=" * 80)
    print(
        "PREPROCESSING TIME BY REPEAT"
    )
    print("=" * 80)

    repeat_summary = (
        timing_df
        .groupby("Repeat")[
            "Preprocessing Time (ms)"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )


    for repeat, row in repeat_summary.iterrows():

        print(
            f"\nRepeat {repeat}:"
        )

        print(
            f"  Mean:   "
            f"{row['mean']:.4f} ms"
        )

        print(
            f"  Median: "
            f"{row['median']:.4f} ms"
        )

        print(
            f"  Std:    "
            f"{row['std']:.4f} ms"
        )

        print(
            f"  Min:    "
            f"{row['min']:.4f} ms"
        )

        print(
            f"  Max:    "
            f"{row['max']:.4f} ms"
        )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    summary_df
):

    row = summary_df.iloc[0]


    print()
    print("=" * 80)
    print(
        "TECHNIQUE B COMPUTATIONAL "
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
        "Final Technique B parameters:"
    )

    print(
        f"  Canny thresholds: "
        f"{row['Canny Thresholds']}"
    )

    print(
        f"  Hough threshold: "
        f"{int(row['Hough Threshold'])}"
    )

    print(
        f"  Min line length: "
        f"{int(row['Min Line Length'])}"
    )

    print(
        f"  Max line gap: "
        f"{int(row['Max Line Gap'])}"
    )


    print()
    print(
        "Custom Technique B preprocessing:"
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

    print(
        f"  Minimum: "
        f"{row['Minimum Custom Preprocessing Time (ms/image)']:.4f} "
        f"ms/image"
    )

    print(
        f"  Maximum: "
        f"{row['Maximum Custom Preprocessing Time (ms/image)']:.4f} "
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
        f"Final test mAP50-95: "
        f"{row['Final Test mAP50-95']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "STUDENT 2 - TECHNIQUE B "
        "PROCESSING TIME ANALYSIS"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Locate test dataset
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


    if len(image_paths) != 1068:

        print(
            "\nWarning: Expected 1068 test "
            f"images, but found "
            f"{len(image_paths)}."
        )


    # --------------------------------------------------------
    # Warm up OpenCV
    # --------------------------------------------------------

    warm_up(
        image_paths
    )


    # --------------------------------------------------------
    # Measure Technique B preprocessing
    # --------------------------------------------------------

    timing_df = (
        measure_preprocessing_time(
            image_paths
        )
    )


    # --------------------------------------------------------
    # Read existing final YOLO metrics
    # --------------------------------------------------------

    yolo_metrics = (
        read_yolo_metrics()
    )


    # --------------------------------------------------------
    # Create computational-efficiency summary
    # --------------------------------------------------------

    summary_df = (
        create_summary(
            timing_df,
            len(image_paths),
            yolo_metrics
        )
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    (
        per_image_path,
        summary_path
    ) = save_results(
        timing_df,
        summary_df
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_repeat_statistics(
        timing_df
    )

    print_summary(
        summary_df
    )


    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FILES GENERATED"
    )
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