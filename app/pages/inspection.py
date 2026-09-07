import hashlib

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.calibration import (
    calibrate_image,
    calculate_bbox_physical_size,
    detect_pcb_corners,
    draw_detected_corners,
)

from modules.preprocessing import (
    apply_technique_d,
    get_technique_d_configuration,
)

from modules.reporting import (
    generate_inspection_pdf,
)

from modules.detection import (
    detect_defects,
    draw_detections,
    summarise_detections,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Inspection",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_inspection" not in st.session_state:
    st.session_state.last_inspection = None

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "pcb_boundary_detection" not in st.session_state:
    st.session_state.pcb_boundary_detection = None

if "pcb_boundary_source" not in st.session_state:
    st.session_state.pcb_boundary_source = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_uploaded_image(uploaded_file):
    """
    Convert a Streamlit uploaded image into an OpenCV
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
    Convert an OpenCV BGR image into RGB format
    for Streamlit display.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def calculate_average_confidence(detections):
    """
    Calculate the mean confidence score across
    all detected defects.
    """

    if not detections:
        return 0.0

    return sum(
        detection["confidence"]
        for detection in detections
    ) / len(detections)


def find_most_common_defect(summary):
    """
    Determine the most frequently detected defect class.
    """

    detected_classes = {
        class_name: count
        for class_name, count
        in summary.items()
        if count > 0
    }

    if not detected_classes:
        return "None"

    return max(
        detected_classes,
        key=detected_classes.get,
    )


def format_defect_name(defect_name):
    """
    Convert internal defect names into readable labels.

    Example:
        missing_hole -> Missing Hole
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


def create_detection_dataframe(
    detections,
    mm_per_pixel=None,
):
    """
    Convert detection results into a table.

    When physical calibration is available,
    approximate bounding-box dimensions are also
    provided in millimetres.
    """

    rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        row = {
            "No.":
                index,

            "Defect Class":
                format_defect_name(
                    detection["class_name"]
                ),

            "Confidence (%)":
                round(
                    detection["confidence"]
                    * 100,
                    2,
                ),

            "X1":
                round(
                    detection["x1"],
                    1,
                ),

            "Y1":
                round(
                    detection["y1"],
                    1,
                ),

            "X2":
                round(
                    detection["x2"],
                    1,
                ),

            "Y2":
                round(
                    detection["y2"],
                    1,
                ),

            "Box Width (px)":
                round(
                    detection["width"],
                    1,
                ),

            "Box Height (px)":
                round(
                    detection["height"],
                    1,
                ),
        }


        # ----------------------------------------------------
        # Optional physical bounding-box measurement
        # ----------------------------------------------------

        if mm_per_pixel is not None:

            physical_size = (
                calculate_bbox_physical_size(
                    detection,
                    mm_per_pixel,
                )
            )

            row[
                "Approx. Box Width (mm)"
            ] = round(
                physical_size["width_mm"],
                2,
            )

            row[
                "Approx. Box Height (mm)"
            ] = round(
                physical_size["height_mm"],
                2,
            )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "PCB Inspection"
)

st.caption(
    "Automated PCB defect inspection using image calibration, "
    "Technique D preprocessing and the trained YOLOv8n model."
)

st.divider()


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.subheader(
    "1. Upload PCB Image"
)


uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
    ],
    help=(
        "Supported formats: JPG, JPEG, PNG and BMP."
    ),
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        original_image = (
            decode_uploaded_image(
                uploaded_file
            )
        )

    except ValueError as error:

        st.error(
            str(error)
        )

        st.stop()


    # ========================================================
    # IMAGE SIGNATURE
    # ========================================================
    #
    # Used to ensure that PCB boundary information from an
    # earlier uploaded image is not accidentally reused.
    # ========================================================

    image_signature = (
        hashlib.md5(
            uploaded_file.getvalue()
        )
        .hexdigest()
    )


    if (
        st.session_state[
            "pcb_boundary_source"
        ]
        != image_signature
    ):

        st.session_state[
            "pcb_boundary_source"
        ] = image_signature

        st.session_state[
            "pcb_boundary_detection"
        ] = None


    image_height, image_width = (
        original_image.shape[:2]
    )


    st.success(
        "Image uploaded successfully."
    )


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    info_col_1, info_col_2 = (
        st.columns(2)
    )


    with info_col_1:

        st.write(
            f"**File Name:** "
            f"{uploaded_file.name}"
        )


    with info_col_2:

        st.write(
            f"**Original Dimensions:** "
            f"{image_width} × "
            f"{image_height} pixels"
        )


    # ========================================================
    # ORIGINAL IMAGE PREVIEW
    # ========================================================

    st.image(
        bgr_to_rgb(
            original_image
        ),
        caption="Original PCB Image",
        use_container_width=True,
    )


    st.divider()


    # ========================================================
    # 2. IMAGE CALIBRATION
    # ========================================================

    st.subheader(
        "2. Image Calibration"
    )


    st.write(
        "Perspective rectification is optional and is mainly "
        "intended for PCB images captured from an angled "
        "viewpoint. If automatic PCB boundary detection fails, "
        "the original image will automatically be used."
    )


    # ========================================================
    # PERSPECTIVE RECTIFICATION OPTION
    # ========================================================

    apply_rectification = (
        st.checkbox(
            "Enable Perspective Rectification",
            value=False,
            help=(
                "The system will attempt to automatically "
                "detect the outer PCB boundary. If detection "
                "fails, inspection continues using the "
                "original image."
            ),
        )
    )


    boundary_detection = None


    # ========================================================
    # AUTOMATIC PCB BOUNDARY DETECTION
    # ========================================================

    if apply_rectification:

        st.markdown(
            "#### Automatic PCB Boundary Detection"
        )


        st.write(
            "The system uses classical image processing "
            "to identify the PCB boundary and estimate "
            "its four corner points."
        )


        detect_boundary_button = (
            st.button(
                "Automatically Detect PCB Boundary",
                use_container_width=True,
            )
        )


        # ----------------------------------------------------
        # RUN AUTOMATIC DETECTION
        # ----------------------------------------------------

        if detect_boundary_button:

            with st.spinner(
                "Detecting PCB boundary..."
            ):

                boundary_detection = (
                    detect_pcb_corners(
                        original_image
                    )
                )


                st.session_state[
                    "pcb_boundary_detection"
                ] = (
                    boundary_detection
                )


        # ----------------------------------------------------
        # LOAD EXISTING RESULT FROM SESSION
        # ----------------------------------------------------

        boundary_detection = (
            st.session_state[
                "pcb_boundary_detection"
            ]
        )


        # ====================================================
        # DISPLAY DETECTION RESULT
        # ====================================================

        if boundary_detection is not None:

            if boundary_detection[
                "success"
            ]:

                st.success(
                    boundary_detection[
                        "message"
                    ]
                )


                # --------------------------------------------
                # DRAW DETECTED PCB BOUNDARY
                # --------------------------------------------

                boundary_preview = (
                    draw_detected_corners(
                        original_image,
                        boundary_detection[
                            "points"
                        ],
                    )
                )


                st.image(
                    bgr_to_rgb(
                        boundary_preview
                    ),
                    caption=(
                        "Automatically Detected "
                        "PCB Boundary"
                    ),
                    use_container_width=True,
                )


                # --------------------------------------------
                # DETECTION INFORMATION
                # --------------------------------------------

                boundary_col_1, boundary_col_2, boundary_col_3 = (
                    st.columns(3)
                )


                boundary_col_1.metric(
                    "Detection Method",
                    boundary_detection[
                        "method"
                    ],
                )


                boundary_col_2.metric(
                    "Image Area Covered",
                    (
                        f"{boundary_detection['area_ratio']:.1%}"
                    ),
                )


                boundary_col_3.metric(
                    "Boundary Score",
                    (
                        f"{boundary_detection['score']:.3f}"
                    ),
                )


                # --------------------------------------------
                # OPTIONAL EDGE MAP
                # --------------------------------------------

                with st.expander(
                    "View Boundary Detection Edge Map"
                ):

                    st.image(
                        boundary_detection[
                            "edge_image"
                        ],
                        caption=(
                            "Edge Map Used for "
                            "PCB Boundary Detection"
                        ),
                        use_container_width=True,
                        clamp=True,
                    )


            else:

                st.warning(
                    "Automatic PCB boundary detection "
                    "was unsuccessful."
                )


                st.info(
                    "No action is required. "
                    "The original PCB image will automatically "
                    "be used for inspection."
                )


    # ========================================================
    # PHYSICAL SCALE CALIBRATION
    # ========================================================

    st.markdown(
        "#### Physical Scale Calibration"
    )


    apply_spatial_scale = (
        st.checkbox(
            "I know the physical PCB width",
            value=False,
            help=(
                "Provide the actual PCB width when physical "
                "bounding-box measurements are required."
            ),
        )
    )


    pcb_width_mm = None


    if apply_spatial_scale:

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
            "The physical width is used to calculate "
            "millimetres per pixel. Any dimensions reported "
            "later represent approximate YOLO bounding-box "
            "dimensions, not exact physical defect dimensions."
        )


    st.divider()


    # ========================================================
    # 3. RUN INSPECTION
    # ========================================================

    st.subheader(
        "3. Run Inspection"
    )


    run_inspection = (
        st.button(
            "Run Inspection",
            type="primary",
            use_container_width=True,
        )
    )


    # ========================================================
    # COMPLETE INSPECTION PIPELINE
    # ========================================================

    if run_inspection:

        st.session_state[
            "last_inspection"
        ] = None

        try:

            with st.spinner(
                "Running PCB inspection..."
            ):


                # ============================================
                # STEP 1:
                # DETERMINE PERSPECTIVE CALIBRATION
                # ============================================

                corner_points = None

                calibration_fallback_used = False

                boundary_method = (
                    "Not Applied"
                )


                if apply_rectification:

                    boundary_detection = (
                        st.session_state[
                            "pcb_boundary_detection"
                        ]
                    )


                    # ----------------------------------------
                    # Automatically attempt boundary
                    # detection if it has not already been
                    # performed successfully.
                    # ----------------------------------------

                    if (
                        boundary_detection is None
                        or not boundary_detection[
                            "success"
                        ]
                    ):

                        boundary_detection = (
                            detect_pcb_corners(
                                original_image
                            )
                        )


                        st.session_state[
                            "pcb_boundary_detection"
                        ] = (
                            boundary_detection
                        )


                    # ----------------------------------------
                    # SUCCESS:
                    # use detected corners
                    # ----------------------------------------

                    if boundary_detection[
                        "success"
                    ]:

                        corner_points = (
                            boundary_detection[
                                "points"
                            ]
                        )

                        boundary_method = (
                            boundary_detection[
                                "method"
                            ]
                        )


                    # ----------------------------------------
                    # FAILURE:
                    # use original image
                    # ----------------------------------------

                    else:

                        corner_points = None

                        calibration_fallback_used = (
                            True
                        )

                        boundary_method = (
                            "Automatic Detection Failed"
                        )


                # ============================================
                # STEP 2:
                # IMAGE CALIBRATION
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

                    calibration_fallback_used = True

                    boundary_method = (
                        "Calibration Failed - Original Used"
                    )

                    calibration = (
                        calibrate_image(
                            original_image,
                            corner_points=None,
                            pcb_width_mm=pcb_width_mm,
                        )
                    )

                calibrated_image = (
                    calibration[
                        "image"
                    ]
                )


                # ============================================
                # STEP 3:
                # TECHNIQUE D PREPROCESSING
                # ============================================

                processed_image = (
                    apply_technique_d(
                        calibrated_image
                    )
                )


                # ============================================
                # STEP 4:
                # YOLOv8n DEFECT DETECTION
                # ============================================

                detections, _ = (
                    detect_defects(
                        processed_image
                    )
                )


                # ============================================
                # STEP 5:
                # DRAW DETECTIONS
                # ============================================

                annotated_image = (
                    draw_detections(
                        calibrated_image,
                        detections,
                    )
                )


                # ============================================
                # STEP 6:
                # DEFECT SUMMARY
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
                # STEP 7:
                # STORE CURRENT INSPECTION
                # ============================================

                inspection_result = {
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

                    "calibration":
                        calibration,

                    "rectification_requested":
                        apply_rectification,

                    "calibration_fallback_used":
                        calibration_fallback_used,

                    "boundary_method":
                        boundary_method,
                }


                st.session_state[
                    "last_inspection"
                ] = (
                    inspection_result
                )


                # ============================================
                # STEP 8:
                # ADD TO DASHBOARD HISTORY
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


            st.success(
                "PCB inspection completed successfully."
            )


        except Exception as error:

            st.error(
                f"Inspection failed: {error}"
            )


# ============================================================
# DISPLAY CURRENT INSPECTION RESULT
# ============================================================

inspection = (
    st.session_state[
        "last_inspection"
    ]
)


if inspection is not None:

    st.divider()


    st.subheader(
        "4. Inspection Result"
    )


    # ========================================================
    # CALIBRATION FALLBACK WARNING
    # ========================================================

    if inspection.get(
        "calibration_fallback_used",
        False,
    ):

        st.warning(
            "Automatic PCB boundary detection was "
            "unsuccessful. The original image was used "
            "for inspection without perspective rectification."
        )


    # ========================================================
    # INSPECTION STATUS
    # ========================================================

    if inspection[
        "total_defects"
    ] > 0:

        st.error(
            f"{inspection['total_defects']} "
            f"PCB defect(s) detected."
        )


    else:

        st.success(
            "No PCB defects were detected."
        )


    # ========================================================
    # KPI CARDS
    # ========================================================

    metric_1, metric_2, metric_3 = (
        st.columns(3)
    )


    metric_1.metric(
        "Total Defects",
        inspection[
            "total_defects"
        ],
    )


    metric_2.metric(
        "Most Common Defect",
        format_defect_name(
            inspection[
                "most_common_defect"
            ]
        ),
    )


    metric_3.metric(
        "Average Confidence",
        (
            f"{inspection['average_confidence']:.2%}"
            if inspection[
                "total_defects"
            ] > 0
            else "N/A"
        ),
    )


    st.divider()


    # ========================================================
    # CALIBRATION INFORMATION
    # ========================================================

    st.subheader(
        "Calibration Information"
    )


    calibration = (
        inspection[
            "calibration"
        ]
    )


    calibration_col_1, calibration_col_2, calibration_col_3, calibration_col_4 = (
        st.columns(4)
    )


    # --------------------------------------------------------
    # RECTIFICATION
    # --------------------------------------------------------

    calibration_col_1.metric(
        "Perspective Rectification",
        (
            "Applied"
            if calibration[
                "rectification_applied"
            ]
            else "Not Applied"
        ),
    )


    # --------------------------------------------------------
    # CALIBRATION METHOD
    # --------------------------------------------------------

    calibration_col_2.metric(
        "Boundary Method",
        inspection.get(
            "boundary_method",
            "Not Applied",
        ),
    )


    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    calibration_col_3.metric(
        "Inspection Dimensions",
        (
            f"{calibration['calibrated_width']} × "
            f"{calibration['calibrated_height']} px"
        ),
    )


    # --------------------------------------------------------
    # SPATIAL SCALE
    # --------------------------------------------------------

    calibration_col_4.metric(
        "Spatial Scale",
        (
            f"{calibration['mm_per_pixel']:.4f} mm/px"
            if calibration[
                "mm_per_pixel"
            ] is not None
            else "Not Provided"
        ),
    )


    # ========================================================
    # CALIBRATION STATUS MESSAGE
    # ========================================================

    if inspection[
        "rectification_requested"
    ]:

        if calibration[
            "rectification_applied"
        ]:

            st.success(
                "Automatic PCB boundary detection succeeded "
                "and perspective rectification was applied."
            )

        else:

            st.info(
                "Perspective rectification was requested, "
                "but automatic boundary detection failed. "
                "Inspection continued using the original image."
            )


    else:

        st.info(
            "Perspective rectification was not requested. "
            "The original image geometry was retained."
        )


    st.divider()


    # ========================================================
    # 5. INSPECTION VISUALISATION
    # ========================================================

    st.subheader(
        "5. Inspection Visualisation"
    )


    original_column, detection_column = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # ORIGINAL PCB IMAGE
    # --------------------------------------------------------

    with original_column:

        st.image(
            bgr_to_rgb(
                inspection[
                    "original_image"
                ]
            ),
            caption="Original PCB Image",
            use_container_width=True,
        )


    # --------------------------------------------------------
    # FINAL DETECTION RESULT
    # --------------------------------------------------------

    with detection_column:

        st.image(
            bgr_to_rgb(
                inspection[
                    "annotated_image"
                ]
            ),
            caption="Detected PCB Defects",
            use_container_width=True,
        )


    # ========================================================
    # RECTIFIED IMAGE
    # ========================================================

    if calibration[
        "rectification_applied"
    ]:

        with st.expander(
            "View Perspective-Rectified Image"
        ):

            st.image(
                bgr_to_rgb(
                    inspection[
                        "calibrated_image"
                    ]
                ),
                caption=(
                    "Perspective-Rectified PCB Image"
                ),
                use_container_width=True,
            )


    # ========================================================
    # TECHNIQUE D OUTPUT
    # ========================================================

    with st.expander(
        "View Technique D Processed Image"
    ):

        st.image(
            bgr_to_rgb(
                inspection[
                    "processed_image"
                ]
            ),
            caption=(
                "Technique D Preprocessed Image"
            ),
            use_container_width=True,
        )


    st.divider()


    # ========================================================
    # 6. DETECTED DEFECTS
    # ========================================================

    st.subheader(
        "6. Detected Defects"
    )


    if inspection[
        "detections"
    ]:

        detection_dataframe = (
            create_detection_dataframe(
                inspection[
                    "detections"
                ],
                mm_per_pixel=(
                    calibration[
                        "mm_per_pixel"
                    ]
                ),
            )
        )


        st.dataframe(
            detection_dataframe,
            use_container_width=True,
            hide_index=True,
        )


        if calibration[
            "mm_per_pixel"
        ] is not None:

            st.caption(
                "Measurements in millimetres represent "
                "approximate YOLO bounding-box dimensions "
                "and should not be interpreted as exact "
                "physical defect dimensions."
            )


    else:

        st.info(
            "No bounding-box detections were "
            "produced for this PCB image."
        )


    st.divider()


    # ========================================================
    # 7. DEFECT SUMMARY
    # ========================================================

    st.subheader(
        "7. Defect Summary"
    )


    summary_rows = []


    for (
        class_name,
        count,
    ) in (
        inspection[
            "defect_summary"
        ].items()
    ):

        summary_rows.append(
            {
                "Defect Class":
                    format_defect_name(
                        class_name
                    ),

                "Detected Count":
                    count,
            }
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


    # ========================================================
    # DEFECT DISTRIBUTION CHART
    # ========================================================

    detected_summary = (
        summary_dataframe[
            summary_dataframe[
                "Detected Count"
            ] > 0
        ]
    )


    if not detected_summary.empty:

        st.markdown(
            "**Detected Defect Distribution**"
        )


        chart_data = (
            detected_summary
            .set_index(
                "Defect Class"
            )[
                "Detected Count"
            ]
        )


        st.bar_chart(
            chart_data
        )


    st.divider()


    # ============================================================
    # 8. EXPORT INSPECTION RESULTS
    # ============================================================

    st.subheader(
        "8. Export Inspection Results"
    )


    export_col_1, export_col_2 = (
        st.columns(2)
    )


    # ============================================================
    # ANNOTATED IMAGE EXPORT
    # ============================================================

    encoding_success, encoded_image = (
        cv2.imencode(
            ".jpg",
            inspection[
                "annotated_image"
            ],
        )
    )


    base_name = (
        inspection[
            "file_name"
        ]
        .rsplit(
            ".",
            1,
        )[0]
    )


    if encoding_success:

        with export_col_1:

            st.download_button(
                label=(
                    "Download Annotated Image"
                ),

                data=(
                    encoded_image
                    .tobytes()
                ),

                file_name=(
                    f"{base_name}"
                    f"_inspection_result.jpg"
                ),

                mime="image/jpeg",

                use_container_width=True,
            )


    # ============================================================
    # PDF REPORT EXPORT
    # ============================================================

    try:

        technique_configuration = (
            get_technique_d_configuration()
        )


        pdf_report = (
            generate_inspection_pdf(
                inspection,
                technique_configuration,
            )
        )


        with export_col_2:

            st.download_button(
                label=(
                    "Download PDF Inspection Report"
                ),

                data=(
                    pdf_report
                ),

                file_name=(
                    f"{base_name}"
                    f"_inspection_report.pdf"
                ),

                mime="application/pdf",

                use_container_width=True,
            )


    except Exception as error:

        with export_col_2:

            st.error(
                f"PDF report generation failed: {error}"
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload a PCB image and select "
        "**Run Inspection** to begin."
    )