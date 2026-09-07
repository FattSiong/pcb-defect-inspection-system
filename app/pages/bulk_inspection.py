import io
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.calibration import (
    calibrate_image,
    detect_pcb_corners,
)

from modules.preprocessing import (
    apply_technique_d,
    get_technique_d_configuration,
)

from modules.detection import (
    detect_defects,
    draw_detections,
    summarise_detections,
)

from modules.reporting import (
    generate_bulk_inspection_pdf,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bulk PCB Inspection",
    page_icon="📁",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_uploaded_image(uploaded_file):
    """
    Convert a Streamlit uploaded file into an OpenCV
    three-channel BGR image.
    """

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.getvalue()
        ),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file could not be read as an image."
        )

    return image


def bgr_to_rgb(image):
    """
    Convert OpenCV BGR image to RGB for Streamlit.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def format_defect_name(defect_name):
    """
    Convert dataset class names into readable labels.
    """

    if defect_name is None:
        return "None"

    if str(defect_name).lower() == "none":
        return "None"

    return (
        str(defect_name)
        .replace("_", " ")
        .title()
    )


def calculate_average_confidence(detections):
    """
    Calculate average confidence across detections.
    """

    if not detections:
        return 0.0

    return sum(
        detection["confidence"]
        for detection in detections
    ) / len(detections)


def find_most_common_defect(defect_summary):
    """
    Determine the most frequently detected defect.
    """

    detected = {
        class_name: count
        for class_name, count
        in defect_summary.items()
        if count > 0
    }

    if not detected:
        return "None"

    return max(
        detected,
        key=detected.get,
    )


def create_safe_output_name(uploaded_name):
    """
    Convert a file name or relative folder path into
    a safe output name.

    Example:
        missing_hole/image01.jpg

    becomes:
        missing_hole__image01
    """

    safe_name = (
        str(uploaded_name)
        .replace("\\", "__")
        .replace("/", "__")
    )

    if "." in safe_name:

        safe_name = (
            safe_name
            .rsplit(
                ".",
                1,
            )[0]
        )

    return safe_name


def encode_jpeg(image):
    """
    Encode an OpenCV image into JPEG bytes.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            "Failed to encode annotated image."
        )

    return encoded.tobytes()


def create_results_zip(results):
    """
    Create an in-memory ZIP containing all annotated
    images from successful inspections.
    """

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for index, result in enumerate(
            results,
            start=1,
        ):

            original_name = (
                result["file_name"]
            )

            safe_name = (
                create_safe_output_name(
                    original_name
                )
            )

            output_name = (
                f"{index:03d}_"
                f"{safe_name}"
                f"_inspection_result.jpg"
            )

            image_bytes = (
                encode_jpeg(
                    result[
                        "annotated_image"
                    ]
                )
            )

            zip_file.writestr(
                output_name,
                image_bytes,
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Bulk PCB Inspection"
)

st.caption(
    "Inspect multiple PCB images or an entire image folder "
    "using Technique D preprocessing and the trained YOLOv8n "
    "defect detection model."
)

st.divider()


# ============================================================
# 1. IMAGE SELECTION
# ============================================================

st.subheader(
    "1. Select PCB Images"
)

st.write(
    "Choose multiple PCB image files individually or "
    "select an entire folder containing PCB images."
)


# ============================================================
# UPLOAD METHOD
# ============================================================

upload_method = st.radio(
    "Upload Method",
    [
        "Upload Images",
        "Upload Folder",
    ],
    horizontal=True,
)


uploaded_files = []


# ============================================================
# INDIVIDUAL IMAGE UPLOAD
# ============================================================

if upload_method == "Upload Images":

    uploaded_files = st.file_uploader(
        "Choose one or more PCB images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ],
        accept_multiple_files=True,
        key="bulk_individual_upload",
        help=(
            "Select multiple PCB images for "
            "batch inspection."
        ),
    )


# ============================================================
# FOLDER UPLOAD
# ============================================================

else:

    uploaded_files = st.file_uploader(
        "Choose a folder containing PCB images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ],
        accept_multiple_files="directory",
        key="bulk_folder_upload",
        help=(
            "Select a folder containing PCB images. "
            "Supported images in subfolders will also "
            "be included."
        ),
    )


# ============================================================
# SELECTED FILE INFORMATION
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} image(s) selected."
    )


    preview_rows = []


    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):

        file_size_kb = (
            uploaded_file.size
            / 1024
        )


        preview_rows.append(
            {
                "No.":
                    index,

                "File Name":
                    uploaded_file.name,

                "File Size":
                    f"{file_size_kb:.1f} KB",
            }
        )


    preview_dataframe = (
        pd.DataFrame(
            preview_rows
        )
    )


    st.dataframe(
        preview_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    if (
        upload_method
        == "Upload Folder"
    ):

        st.info(
            "All supported PCB images found in the selected "
            "folder are ready for bulk inspection."
        )


    st.divider()


    # ========================================================
    # 2. CALIBRATION OPTIONS
    # ========================================================

    st.subheader(
        "2. Calibration Options"
    )


    st.write(
        "The following calibration settings will be "
        "applied independently to every image in the batch."
    )


    # ========================================================
    # AUTOMATIC RECTIFICATION
    # ========================================================

    auto_rectification = (
        st.checkbox(
            "Attempt Automatic Perspective Rectification",
            value=False,
            help=(
                "The system attempts to detect the PCB "
                "boundary for each image. If boundary "
                "detection or rectification fails, the "
                "original image is used automatically."
            ),
        )
    )


    if auto_rectification:

        st.caption(
            "Calibration failure will not stop the batch. "
            "The affected image will continue through "
            "Technique D and YOLO using its original geometry."
        )


    # ========================================================
    # PHYSICAL SCALE
    # ========================================================

    use_physical_width = (
        st.checkbox(
            "Use a Common Physical PCB Width",
            value=False,
            help=(
                "Enable this only when every PCB in the "
                "selected batch has the same known width."
            ),
        )
    )


    pcb_width_mm = None


    if use_physical_width:

        pcb_width_mm = (
            st.number_input(
                "Physical PCB Width (mm)",
                min_value=0.1,
                value=100.0,
                step=0.1,
                format="%.2f",
            )
        )


        st.caption(
            "The same physical PCB width will be used "
            "for all images in this batch."
        )


    st.divider()


    # ========================================================
    # 3. RUN BULK INSPECTION
    # ========================================================

    st.subheader(
        "3. Run Bulk Inspection"
    )


    run_bulk_inspection = (
        st.button(
            "Run Bulk Inspection",
            type="primary",
            use_container_width=True,
        )
    )


    # ========================================================
    # BULK PROCESSING PIPELINE
    # ========================================================

    if run_bulk_inspection:

        # Clear results from the previous bulk run.
        st.session_state.bulk_results = []


        total_files = len(
            uploaded_files
        )


        progress_bar = (
            st.progress(
                0
            )
        )


        progress_text = (
            st.empty()
        )


        # ====================================================
        # PROCESS EACH IMAGE
        # ====================================================

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            progress_text.write(
                f"Processing {index} of {total_files}: "
                f"**{uploaded_file.name}**"
            )


            try:

                # ============================================
                # STEP 1:
                # LOAD ORIGINAL IMAGE
                # ============================================

                original_image = (
                    decode_uploaded_image(
                        uploaded_file
                    )
                )


                # ============================================
                # STEP 2:
                # AUTOMATIC PCB BOUNDARY DETECTION
                # ============================================

                corner_points = None

                boundary_detection_method = (
                    "Not Applied"
                )

                calibration_fallback_used = (
                    False
                )


                if auto_rectification:

                    try:

                        boundary_detection = (
                            detect_pcb_corners(
                                original_image
                            )
                        )


                        if boundary_detection[
                            "success"
                        ]:

                            corner_points = (
                                boundary_detection[
                                    "points"
                                ]
                            )

                            boundary_detection_method = (
                                boundary_detection[
                                    "method"
                                ]
                            )


                        else:

                            # --------------------------------
                            # Boundary detection failed.
                            # Continue with original image.
                            # --------------------------------

                            corner_points = None

                            calibration_fallback_used = (
                                True
                            )

                            boundary_detection_method = (
                                "Automatic Detection Failed"
                            )


                    except Exception:

                        corner_points = None

                        calibration_fallback_used = (
                            True
                        )

                        boundary_detection_method = (
                            "Automatic Detection Failed"
                        )


                # ============================================
                # STEP 3:
                # CALIBRATION WITH SAFE FALLBACK
                # ============================================

                try:

                    calibration = (
                        calibrate_image(
                            original_image,
                            corner_points=(
                                corner_points
                            ),
                            pcb_width_mm=(
                                pcb_width_mm
                            ),
                        )
                    )


                except (
                    ValueError,
                    cv2.error,
                ):

                    # ----------------------------------------
                    # Perspective transformation was not safe.
                    # Continue using the original image.
                    # ----------------------------------------

                    calibration_fallback_used = (
                        True
                    )

                    boundary_detection_method = (
                        "Calibration Failed - Original Used"
                    )


                    calibration = (
                        calibrate_image(
                            original_image,
                            corner_points=None,
                            pcb_width_mm=(
                                pcb_width_mm
                            ),
                        )
                    )


                calibrated_image = (
                    calibration[
                        "image"
                    ]
                )


                # ============================================
                # STEP 4:
                # TECHNIQUE D PREPROCESSING
                # ============================================

                processed_image = (
                    apply_technique_d(
                        calibrated_image
                    )
                )


                # ============================================
                # STEP 5:
                # YOLOv8n DETECTION
                # ============================================

                detections, _ = (
                    detect_defects(
                        processed_image
                    )
                )


                # ============================================
                # STEP 6:
                # DRAW DETECTIONS
                # ============================================

                annotated_image = (
                    draw_detections(
                        calibrated_image,
                        detections,
                    )
                )


                # ============================================
                # STEP 7:
                # DEFECT ANALYSIS
                # ============================================

                defect_summary = (
                    summarise_detections(
                        detections
                    )
                )


                total_defects = len(
                    detections
                )


                average_confidence = (
                    calculate_average_confidence(
                        detections
                    )
                )


                most_common_defect = (
                    find_most_common_defect(
                        defect_summary
                    )
                )


                # ============================================
                # STEP 8:
                # STORE BULK RESULT
                # ============================================

                result = {
                    "file_name":
                        uploaded_file.name,

                    "original_image":
                        original_image,

                    "calibrated_image":
                        calibrated_image,

                    "processed_image":
                        processed_image,

                    "annotated_image":
                        annotated_image,

                    "detections":
                        detections,

                    "defect_summary":
                        defect_summary,

                    "total_defects":
                        total_defects,

                    "average_confidence":
                        average_confidence,

                    "most_common_defect":
                        most_common_defect,

                    "rectification_requested":
                        auto_rectification,

                    "rectification_applied":
                        calibration[
                            "rectification_applied"
                        ],

                    "calibration_fallback_used":
                        calibration_fallback_used,

                    "boundary_method":
                        boundary_detection_method,

                    "spatial_calibration":
                        (
                            calibration[
                                "mm_per_pixel"
                            ]
                            is not None
                        ),

                    "status":
                        "Success",
                }


                st.session_state.bulk_results.append(
                    result
                )


                # ============================================
                # STEP 9:
                # ADD RESULT TO DASHBOARD HISTORY
                # ============================================

                history_record = {
                    "File Name":
                        uploaded_file.name,

                    "Total Defects":
                        total_defects,

                    "Most Common Defect":
                        format_defect_name(
                            most_common_defect
                        ),

                    "Average Confidence":
                        round(
                            average_confidence,
                            4,
                        ),

                    "Defect Summary":
                        defect_summary.copy(),

                    "Rectification Applied":
                        calibration[
                            "rectification_applied"
                        ],

                    "Calibration Fallback":
                        calibration_fallback_used,

                    "Spatial Calibration":
                        (
                            calibration[
                                "mm_per_pixel"
                            ]
                            is not None
                        ),
                }


                st.session_state[
                    "inspection_history"
                ].append(
                    history_record
                )


            # =================================================
            # IMAGE PROCESSING FAILURE
            # =================================================

            except Exception as error:

                st.session_state.bulk_results.append(
                    {
                        "file_name":
                            uploaded_file.name,

                        "total_defects":
                            0,

                        "average_confidence":
                            0.0,

                        "most_common_defect":
                            "None",

                        "rectification_requested":
                            auto_rectification,

                        "rectification_applied":
                            False,

                        "calibration_fallback_used":
                            False,

                        "boundary_method":
                            "Not Available",

                        "spatial_calibration":
                            False,

                        "status":
                            "Failed",

                        "error":
                            str(error),
                    }
                )


            # =================================================
            # UPDATE PROGRESS
            # =================================================

            progress_percentage = int(
                (
                    index
                    / total_files
                )
                * 100
            )


            progress_bar.progress(
                progress_percentage
            )


        # ====================================================
        # COMPLETE PROGRESS DISPLAY
        # ====================================================

        progress_text.empty()


        successful_count = sum(
            1
            for result
            in st.session_state.bulk_results
            if result[
                "status"
            ]
            == "Success"
        )


        failed_count = (
            total_files
            - successful_count
        )


        if failed_count == 0:

            st.success(
                f"Bulk inspection completed successfully "
                f"for all {successful_count} image(s)."
            )


        else:

            st.warning(
                f"Bulk inspection completed. "
                f"{successful_count} image(s) succeeded and "
                f"{failed_count} image(s) failed."
            )


# ============================================================
# BULK RESULTS
# ============================================================

bulk_results = (
    st.session_state.bulk_results
)


if bulk_results:

    st.divider()


    # ========================================================
    # 4. BULK INSPECTION SUMMARY
    # ========================================================

    st.subheader(
        "4. Bulk Inspection Summary"
    )


    successful_results = [
        result
        for result in bulk_results
        if result[
            "status"
        ]
        == "Success"
    ]


    failed_results = [
        result
        for result in bulk_results
        if result[
            "status"
        ]
        == "Failed"
    ]


    total_detected_defects = sum(
        result[
            "total_defects"
        ]
        for result
        in successful_results
    )


    images_with_defects = sum(
        1
        for result
        in successful_results
        if result[
            "total_defects"
        ] > 0
    )


    fallback_count = sum(
        1
        for result
        in successful_results
        if result.get(
            "calibration_fallback_used",
            False,
        )
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    metric_1, metric_2, metric_3, metric_4, metric_5 = (
        st.columns(5)
    )


    metric_1.metric(
        "Images Processed",
        len(
            successful_results
        ),
    )


    metric_2.metric(
        "Total Defects",
        total_detected_defects,
    )


    metric_3.metric(
        "Images with Detections",
        images_with_defects,
    )


    metric_4.metric(
        "Calibration Fallbacks",
        fallback_count,
    )


    metric_5.metric(
        "Failed Images",
        len(
            failed_results
        ),
    )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_rows = []


    for index, result in enumerate(
        bulk_results,
        start=1,
    ):

        total_defects = int(
            result.get(
                "total_defects",
                0,
            )
        )


        average_confidence = float(
            result.get(
                "average_confidence",
                0.0,
            )
        )


        row = {
            "No.":
                index,

            "File Name":
                result[
                    "file_name"
                ],

            "Status":
                result[
                    "status"
                ],

            "Total Defects":
                total_defects,

            "Most Common Defect":
                format_defect_name(
                    result.get(
                        "most_common_defect",
                        "None",
                    )
                ),

            "Average Confidence (%)":
                (
                    round(
                        average_confidence
                        * 100,
                        2,
                    )
                    if total_defects > 0
                    else None
                ),

            "Rectification":
                (
                    "Yes"
                    if result.get(
                        "rectification_applied",
                        False,
                    )
                    else "No"
                ),

            "Calibration Fallback":
                (
                    "Yes"
                    if result.get(
                        "calibration_fallback_used",
                        False,
                    )
                    else "No"
                ),

            "Boundary Method":
                result.get(
                    "boundary_method",
                    "Not Applied",
                ),

            "Spatial Calibration":
                (
                    "Yes"
                    if result.get(
                        "spatial_calibration",
                        False,
                    )
                    else "No"
                ),

            "Error":
                (
                    result.get(
                        "error",
                        ""
                    )
                    if result[
                        "status"
                    ]
                    == "Failed"
                    else ""
                ),
        }


        summary_rows.append(
            row
        )


    summary_dataframe = (
        pd.DataFrame(
            summary_rows
        )
    )


    st.dataframe(
        summary_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    # ========================================================
    # 5. INDIVIDUAL INSPECTION RESULTS
    # ========================================================

    st.subheader(
        "5. Individual Inspection Results"
    )


    for result in bulk_results:

        with st.expander(
            result[
                "file_name"
            ]
        ):

            # =================================================
            # FAILED RESULT
            # =================================================

            if (
                result[
                    "status"
                ]
                == "Failed"
            ):

                st.error(
                    result.get(
                        "error",
                        "Inspection failed.",
                    )
                )

                continue


            # =================================================
            # RESULT METRICS
            # =================================================

            info_1, info_2, info_3 = (
                st.columns(3)
            )


            info_1.metric(
                "Detected Defects",
                result[
                    "total_defects"
                ],
            )


            info_2.metric(
                "Most Common Defect",
                format_defect_name(
                    result[
                        "most_common_defect"
                    ]
                ),
            )


            info_3.metric(
                "Average Confidence",
                (
                    f"{result['average_confidence']:.2%}"
                    if result[
                        "total_defects"
                    ] > 0
                    else "N/A"
                ),
            )


            # =================================================
            # CALIBRATION MESSAGE
            # =================================================

            if result.get(
                "calibration_fallback_used",
                False,
            ):

                st.warning(
                    "Automatic PCB calibration was "
                    "unsuccessful for this image. "
                    "The original image was used instead."
                )


            elif result.get(
                "rectification_applied",
                False,
            ):

                st.success(
                    "Perspective rectification was "
                    "successfully applied."
                )


            # =================================================
            # IMAGE COMPARISON
            # =================================================

            image_col_1, image_col_2 = (
                st.columns(2)
            )


            with image_col_1:

                st.image(
                    bgr_to_rgb(
                        result[
                            "original_image"
                        ]
                    ),
                    caption=(
                        "Original PCB Image"
                    ),
                    use_container_width=True,
                )


            with image_col_2:

                st.image(
                    bgr_to_rgb(
                        result[
                            "annotated_image"
                        ]
                    ),
                    caption=(
                        "Inspection Result"
                    ),
                    use_container_width=True,
                )


            # =================================================
            # TECHNIQUE D IMAGE
            # =================================================

            with st.expander(
                "View Technique D Processed Image"
            ):

                st.image(
                    bgr_to_rgb(
                        result[
                            "processed_image"
                        ]
                    ),
                    caption=(
                        "Technique D "
                        "Preprocessed Image"
                    ),
                    use_container_width=True,
                )


    st.divider()


    # ========================================================
    # 6. EXPORT BULK RESULTS
    # ========================================================

    st.subheader(
        "6. Export Bulk Results"
    )


    export_col_1, export_col_2, export_col_3 = (
        st.columns(3)
    )


    # ========================================================
    # CSV EXPORT
    # ========================================================

    csv_data = (
        summary_dataframe
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    with export_col_1:

        st.download_button(
            label=(
                "Download Summary CSV"
            ),

            data=(
                csv_data
            ),

            file_name=(
                "pcb_bulk_inspection_summary.csv"
            ),

            mime="text/csv",

            use_container_width=True,
        )


    # ========================================================
    # ANNOTATED IMAGES ZIP EXPORT
    # ========================================================

    if successful_results:

        try:

            zip_data = (
                create_results_zip(
                    successful_results
                )
            )


            with export_col_2:

                st.download_button(
                    label=(
                        "Download Annotated Images ZIP"
                    ),

                    data=(
                        zip_data
                    ),

                    file_name=(
                        "pcb_bulk_inspection_results.zip"
                    ),

                    mime="application/zip",

                    use_container_width=True,
                )


        except Exception as error:

            with export_col_2:

                st.error(
                    f"ZIP generation failed: "
                    f"{error}"
                )


    else:

        with export_col_2:

            st.info(
                "No annotated images are "
                "available for download."
            )


    # ========================================================
    # BULK PDF REPORT EXPORT
    # ========================================================

    try:

        technique_configuration = (
            get_technique_d_configuration()
        )


        bulk_pdf_report = (
            generate_bulk_inspection_pdf(
                bulk_results,
                technique_configuration,
            )
        )


        with export_col_3:

            st.download_button(
                label=(
                    "Download Bulk PDF Report"
                ),

                data=(
                    bulk_pdf_report
                ),

                file_name=(
                    "pcb_bulk_inspection_report.pdf"
                ),

                mime="application/pdf",

                use_container_width=True,
            )


    except Exception as error:

        with export_col_3:

            st.error(
                f"Bulk PDF generation failed: "
                f"{error}"
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    if not uploaded_files:

        st.info(
            "Select PCB images or upload a folder "
            "to begin bulk inspection."
        )