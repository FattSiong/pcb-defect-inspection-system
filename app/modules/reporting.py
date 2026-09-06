from io import BytesIO
from datetime import datetime

import cv2

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_defect_name(name):
    """
    Convert an internal class name into a readable label.

    Example:
        missing_hole -> Missing Hole
    """

    if name is None:
        return "None"

    if str(name).lower() == "none":
        return "None"

    return (
        str(name)
        .replace("_", " ")
        .title()
    )


def format_boolean(value):
    """
    Convert Boolean value to Yes / No.
    """

    return (
        "Yes"
        if value
        else "No"
    )


def encode_image_for_pdf(
    image,
):
    """
    Convert an OpenCV BGR image into JPEG data that
    can be inserted directly into a PDF.
    """

    if image is None:
        return None


    # OpenCV stores colour images using BGR.
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


    # ReportLab Image can read encoded JPEG bytes.
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        ),
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            95,
        ],
    )


    if not success:
        raise RuntimeError(
            "Failed to encode image for PDF report."
        )


    image_buffer = BytesIO(
        encoded.tobytes()
    )

    image_buffer.seek(0)

    return image_buffer


def calculate_report_image_size(
    image,
    maximum_width=170 * mm,
    maximum_height=110 * mm,
):
    """
    Calculate an appropriate PDF display size while
    maintaining image aspect ratio.
    """

    height, width = (
        image.shape[:2]
    )


    scale_width = (
        maximum_width
        / width
    )


    scale_height = (
        maximum_height
        / height
    )


    scale = min(
        scale_width,
        scale_height,
    )


    return (
        width * scale,
        height * scale,
    )


# ============================================================
# PAGE NUMBER
# ============================================================

def add_page_number(
    canvas,
    document,
):
    """
    Add a page number to the bottom-right corner.
    """

    page_number = (
        canvas.getPageNumber()
    )


    canvas.saveState()


    canvas.setFont(
        "Helvetica",
        8,
    )


    canvas.drawRightString(
        A4[0] - 15 * mm,
        10 * mm,
        f"Page {page_number}",
    )


    canvas.restoreState()


# ============================================================
# MAIN PDF GENERATOR
# ============================================================

def generate_inspection_pdf(
    inspection,
    technique_configuration,
):
    """
    Generate a complete PCB defect inspection PDF report.

    Parameters
    ----------
    inspection : dict
        Current inspection result stored in
        st.session_state.last_inspection.

    technique_configuration : dict
        Final selected Technique D configuration.

    Returns
    -------
    bytes
        Complete PDF document.
    """

    if inspection is None:

        raise ValueError(
            "Inspection data cannot be None."
        )


    # ========================================================
    # DOCUMENT BUFFER
    # ========================================================

    pdf_buffer = BytesIO()


    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,

        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,

        title=(
            "PCB Defect Inspection Report"
        ),

        author=(
            "PCB Defect Inspection System"
        ),
    )


    # ========================================================
    # STYLES
    # ========================================================

    styles = (
        getSampleStyleSheet()
    )


    title_style = (
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )


    subtitle_style = (
        ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=15,
        )
    )


    heading_style = (
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=8,
            spaceAfter=8,
        )
    )


    body_style = (
        ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
        )
    )


    warning_style = (
        ParagraphStyle(
            "Warning",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.darkred,
        )
    )


    story = []


    # ========================================================
    # REPORT HEADER
    # ========================================================

    story.append(
        Paragraph(
            "PCB Defect Inspection Report",
            title_style,
        )
    )


    story.append(
        Paragraph(
            (
                "Generated by the Image-Based PCB "
                "Defect Inspection System"
            ),
            subtitle_style,
        )
    )


    generation_time = (
        datetime.now()
        .strftime(
            "%d %B %Y, %H:%M:%S"
        )
    )


    # ========================================================
    # 1. INSPECTION INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Inspection Information",
            heading_style,
        )
    )


    file_name = (
        inspection.get(
            "file_name",
            "Unknown",
        )
    )


    original_image = (
        inspection.get(
            "original_image"
        )
    )


    if original_image is not None:

        original_height, original_width = (
            original_image.shape[:2]
        )

        original_dimensions = (
            f"{original_width} × "
            f"{original_height} pixels"
        )

    else:

        original_dimensions = (
            "Not Available"
        )


    information_data = [
        [
            "Item",
            "Details",
        ],

        [
            "File Name",
            file_name,
        ],

        [
            "Original Dimensions",
            original_dimensions,
        ],

        [
            "Report Generated",
            generation_time,
        ],
    ]


    information_table = Table(
        information_data,
        colWidths=[
            55 * mm,
            115 * mm,
        ],
    )


    information_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        information_table
    )


    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )


    # ========================================================
    # 2. INSPECTION SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "2. Inspection Summary",
            heading_style,
        )
    )


    total_defects = int(
        inspection.get(
            "total_defects",
            0,
        )
    )


    most_common_defect = (
        format_defect_name(
            inspection.get(
                "most_common_defect",
                "None",
            )
        )
    )


    average_confidence = float(
        inspection.get(
            "average_confidence",
            0.0,
        )
    )


    average_confidence_display = (
        f"{average_confidence:.2%}"
        if total_defects > 0
        else "N/A"
    )


    inspection_summary = [
        [
            "Metric",
            "Result",
        ],

        [
            "Total Defects Detected",
            str(
                total_defects
            ),
        ],

        [
            "Most Common Defect",
            most_common_defect,
        ],

        [
            "Average Confidence",
            average_confidence_display,
        ],
    ]


    summary_table = Table(
        inspection_summary,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )


    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        summary_table
    )


    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )


    if total_defects > 0:

        story.append(
            Paragraph(
                (
                    f"The system produced "
                    f"{total_defects} PCB defect "
                    f"detection(s) above the configured "
                    f"confidence threshold."
                ),
                body_style,
            )
        )

    else:

        story.append(
            Paragraph(
                (
                    "No PCB defect detections were produced "
                    "above the configured confidence threshold."
                ),
                body_style,
            )
        )


    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )


    story.append(
        Paragraph(
            (
                "Note: 'No defects detected' does not provide "
                "absolute confirmation that the physical PCB "
                "is defect-free. Results are based on the "
                "trained detection model and configured "
                "confidence threshold."
            ),
            warning_style,
        )
    )


    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )


    # ========================================================
    # 3. CALIBRATION INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "3. Calibration Information",
            heading_style,
        )
    )


    calibration = (
        inspection.get(
            "calibration",
            {},
        )
    )


    rectification_applied = bool(
        calibration.get(
            "rectification_applied",
            False,
        )
    )


    fallback_used = bool(
        inspection.get(
            "calibration_fallback_used",
            False,
        )
    )


    boundary_method = (
        inspection.get(
            "boundary_method",
            "Not Applied",
        )
    )


    calibrated_width = (
        calibration.get(
            "calibrated_width",
            "N/A",
        )
    )


    calibrated_height = (
        calibration.get(
            "calibrated_height",
            "N/A",
        )
    )


    mm_per_pixel = (
        calibration.get(
            "mm_per_pixel"
        )
    )


    if mm_per_pixel is None:

        spatial_scale = (
            "Not Provided"
        )

    else:

        spatial_scale = (
            f"{mm_per_pixel:.6f} mm/pixel"
        )


    calibration_data = [
        [
            "Calibration Item",
            "Result",
        ],

        [
            "Perspective Rectification Applied",
            format_boolean(
                rectification_applied
            ),
        ],

        [
            "Automatic Calibration Fallback Used",
            format_boolean(
                fallback_used
            ),
        ],

        [
            "Boundary Detection Method",
            boundary_method,
        ],

        [
            "Inspection Image Dimensions",
            (
                f"{calibrated_width} × "
                f"{calibrated_height} pixels"
            ),
        ],

        [
            "Spatial Scale",
            spatial_scale,
        ],
    ]


    calibration_table = Table(
        calibration_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )


    calibration_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        calibration_table
    )


    if fallback_used:

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        story.append(
            Paragraph(
                (
                    "Automatic PCB boundary detection or "
                    "perspective calibration was unsuitable "
                    "for this image. The original image was "
                    "therefore used for defect inspection."
                ),
                warning_style,
            )
        )


    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )


    # ========================================================
    # 4. DETECTED DEFECT DETAILS
    # ========================================================

    story.append(
        Paragraph(
            "4. Detected Defect Details",
            heading_style,
        )
    )


    detections = (
        inspection.get(
            "detections",
            [],
        )
    )


    if detections:

        detection_data = [
            [
                "No.",
                "Defect",
                "Confidence",
                "Bounding Box (px)",
            ]
        ]


        if mm_per_pixel is not None:

            detection_data[0].append(
                "Approx. Box Size (mm)"
            )


        for index, detection in enumerate(
            detections,
            start=1,
        ):

            row = [
                str(index),

                format_defect_name(
                    detection[
                        "class_name"
                    ]
                ),

                (
                    f"{detection['confidence']:.2%}"
                ),

                (
                    f"{detection['width']:.1f} × "
                    f"{detection['height']:.1f}"
                ),
            ]


            if mm_per_pixel is not None:

                width_mm = (
                    detection[
                        "width"
                    ]
                    * mm_per_pixel
                )


                height_mm = (
                    detection[
                        "height"
                    ]
                    * mm_per_pixel
                )


                row.append(
                    (
                        f"{width_mm:.2f} × "
                        f"{height_mm:.2f}"
                    )
                )


            detection_data.append(
                row
            )


        if mm_per_pixel is not None:

            detection_widths = [
                12 * mm,
                38 * mm,
                28 * mm,
                45 * mm,
                45 * mm,
            ]

        else:

            detection_widths = [
                15 * mm,
                50 * mm,
                35 * mm,
                70 * mm,
            ]


        detection_table = Table(
            detection_data,
            colWidths=detection_widths,
            repeatRows=1,
        )


        detection_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),

                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (0, -1),
                        "CENTER",
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )


        story.append(
            detection_table
        )


        if mm_per_pixel is not None:

            story.append(
                Spacer(
                    1,
                    4 * mm,
                )
            )

            story.append(
                Paragraph(
                    (
                        "Physical dimensions shown above "
                        "represent approximate YOLO "
                        "bounding-box dimensions and should "
                        "not be interpreted as exact physical "
                        "defect dimensions."
                    ),
                    warning_style,
                )
            )


    else:

        story.append(
            Paragraph(
                (
                    "No bounding-box detections were "
                    "produced for this image."
                ),
                body_style,
            )
        )


    # ========================================================
    # NEW PAGE FOR VISUAL RESULT
    # ========================================================

    story.append(
        PageBreak()
    )


    # ========================================================
    # 5. INSPECTION VISUALISATION
    # ========================================================

    story.append(
        Paragraph(
            "5. Inspection Visualisation",
            heading_style,
        )
    )


    annotated_image = (
        inspection.get(
            "annotated_image"
        )
    )


    if annotated_image is not None:

        image_buffer = (
            encode_image_for_pdf(
                annotated_image
            )
        )


        display_width, display_height = (
            calculate_report_image_size(
                annotated_image
            )
        )


        report_image = Image(
            image_buffer,
            width=display_width,
            height=display_height,
        )


        story.append(
            report_image
        )


        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )


        story.append(
            Paragraph(
                (
                    "Figure: PCB inspection result with "
                    "YOLOv8n bounding-box detections."
                ),
                subtitle_style,
            )
        )


    # ========================================================
    # 6. TECHNIQUE D CONFIGURATION
    # ========================================================

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )


    story.append(
        Paragraph(
            "6. Selected Technique D Configuration",
            heading_style,
        )
    )


    configuration_data = [
        [
            Paragraph(
                "<b>Parameter</b>",
                body_style,
            ),
            Paragraph(
                "<b>Selected Value</b>",
                body_style,
            ),
        ],

        [
            "Selected Technique",
            "Technique D",
        ],

        [
            "Processing Method",
            Paragraph(
                (
                    "Bilateral Filtering + Multi-scale "
                    "Top-Hat/Black-Hat Morphological "
                    "Enhancement"
                ),
                body_style,
            ),
        ],

        [
            "Bilateral Diameter",
            "3",
        ],

        [
            "Sigma Colour",
            "15",
        ],

        [
            "Sigma Space",
            "3",
        ],

        [
            "Morphology Kernel Sizes",
            "7 × 7, 15 × 15, 31 × 31",
        ],

        [
            "Morphology Kernel Shape",
            "Ellipse",
        ],

        [
            "Top-Hat Weight (α)",
            "0.7",
        ],

        [
            "Black-Hat Weight (β)",
            "0.7",
        ],
    ]


    configuration_table = Table(
        configuration_data,
        colWidths=[
            70 * mm,
            100 * mm,
        ],
        repeatRows=1,
    )


    configuration_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        configuration_table
    )


    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )


    story.append(
        Paragraph(
            (
                "Technique D applies bilateral filtering to "
                "the LAB lightness channel, followed by "
                "multi-scale Top-Hat and Black-Hat "
                "morphological enhancement using the final "
                "experimentally selected configuration."
            ),
            body_style,
        )
    )

    # ========================================================
    # BUILD DOCUMENT
    # ========================================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


    pdf_buffer.seek(0)


    return (
        pdf_buffer.getvalue()
    )
    
    # ============================================================
# BULK INSPECTION PDF REPORT
# ============================================================

def generate_bulk_inspection_pdf(
    bulk_results,
    technique_configuration,
):
    """
    Generate a combined PDF report for a bulk PCB
    inspection session.

    Parameters
    ----------
    bulk_results : list
        Results produced by bulk_inspection.py.

    technique_configuration : dict
        Final selected Technique D configuration.

    Returns
    -------
    bytes
        Complete PDF document.
    """

    if not bulk_results:
        raise ValueError(
            "Bulk inspection results cannot be empty."
        )


    # ========================================================
    # DOCUMENT BUFFER
    # ========================================================

    pdf_buffer = BytesIO()


    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,

        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,

        title=(
            "Bulk PCB Defect Inspection Report"
        ),

        author=(
            "PCB Defect Inspection System"
        ),
    )


    # ========================================================
    # STYLES
    # ========================================================

    styles = (
        getSampleStyleSheet()
    )


    title_style = ParagraphStyle(
        "BulkReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=10,
    )


    subtitle_style = ParagraphStyle(
        "BulkReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=15,
    )


    heading_style = ParagraphStyle(
        "BulkSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=8,
    )


    body_style = ParagraphStyle(
        "BulkBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )


    warning_style = ParagraphStyle(
        "BulkWarning",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.darkred,
    )


    story = []


    # ========================================================
    # HEADER
    # ========================================================

    story.append(
        Paragraph(
            "Bulk PCB Defect Inspection Report",
            title_style,
        )
    )


    story.append(
        Paragraph(
            (
                "Generated by the Image-Based PCB "
                "Defect Inspection System"
            ),
            subtitle_style,
        )
    )


    generation_time = (
        datetime.now()
        .strftime(
            "%d %B %Y, %H:%M:%S"
        )
    )


    # ========================================================
    # SUMMARY VALUES
    # ========================================================

    successful_results = [
        result
        for result in bulk_results
        if result.get(
            "status"
        ) == "Success"
    ]


    failed_results = [
        result
        for result in bulk_results
        if result.get(
            "status"
        ) == "Failed"
    ]


    total_defects = sum(
        int(
            result.get(
                "total_defects",
                0,
            )
        )
        for result in successful_results
    )


    images_with_detections = sum(
        1
        for result in successful_results
        if int(
            result.get(
                "total_defects",
                0,
            )
        ) > 0
    )


    valid_confidences = [
        float(
            result.get(
                "average_confidence",
                0.0,
            )
        )
        for result in successful_results
        if int(
            result.get(
                "total_defects",
                0,
            )
        ) > 0
    ]


    if valid_confidences:

        overall_average_confidence = (
            sum(
                valid_confidences
            )
            / len(
                valid_confidences
            )
        )

        average_confidence_display = (
            f"{overall_average_confidence:.2%}"
        )

    else:

        average_confidence_display = (
            "N/A"
        )


    # ========================================================
    # 1. BULK SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Bulk Inspection Summary",
            heading_style,
        )
    )


    summary_data = [
        [
            "Metric",
            "Result",
        ],

        [
            "Total Images Selected",
            str(
                len(
                    bulk_results
                )
            ),
        ],

        [
            "Successfully Processed",
            str(
                len(
                    successful_results
                )
            ),
        ],

        [
            "Failed Images",
            str(
                len(
                    failed_results
                )
            ),
        ],

        [
            "Total Defects Detected",
            str(
                total_defects
            ),
        ],

        [
            "Images with Detections",
            str(
                images_with_detections
            ),
        ],

        [
            "Average Confidence",
            average_confidence_display,
        ],

        [
            "Report Generated",
            generation_time,
        ],
    ]


    summary_table = Table(
        summary_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )


    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        summary_table
    )


    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )


    story.append(
        Paragraph(
            (
                "Detection results are based on the trained "
                "YOLOv8n model and the configured confidence "
                "threshold. An image with no detections should "
                "not be interpreted as absolute confirmation "
                "that the physical PCB is defect-free."
            ),
            warning_style,
        )
    )


    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )


    # ========================================================
    # 2. IMAGE-BY-IMAGE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "2. Image-by-Image Inspection Summary",
            heading_style,
        )
    )


    result_table_data = [
        [
            "No.",
            "File Name",
            "Status",
            "Defects",
            "Most Common Defect",
            "Avg. Confidence",
        ]
    ]


    for index, result in enumerate(
        bulk_results,
        start=1,
    ):

        total = int(
            result.get(
                "total_defects",
                0,
            )
        )


        confidence = float(
            result.get(
                "average_confidence",
                0.0,
            )
        )


        confidence_display = (
            f"{confidence:.2%}"
            if total > 0
            else "N/A"
        )


        result_table_data.append(
            [
                str(index),

                Paragraph(
                    str(
                        result.get(
                            "file_name",
                            "Unknown",
                        )
                    ),
                    body_style,
                ),

                result.get(
                    "status",
                    "Unknown",
                ),

                str(
                    total
                ),

                format_defect_name(
                    result.get(
                        "most_common_defect",
                        "None",
                    )
                ),

                confidence_display,
            ]
        )


    result_table = Table(
        result_table_data,
        colWidths=[
            10 * mm,
            50 * mm,
            22 * mm,
            18 * mm,
            42 * mm,
            28 * mm,
        ],
        repeatRows=1,
    )


    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (0, -1),
                    "CENTER",
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )


    story.append(
        result_table
    )


    # ========================================================
    # FAILED IMAGE DETAILS
    # ========================================================

    if failed_results:

        story.append(
            Spacer(
                1,
                7 * mm,
            )
        )


        story.append(
            Paragraph(
                "Failed Image Details",
                heading_style,
            )
        )


        for result in failed_results:

            story.append(
                Paragraph(
                    (
                        f"<b>{result.get('file_name', 'Unknown')}</b>: "
                        f"{result.get('error', 'Unknown processing error')}"
                    ),
                    warning_style,
                )
            )


            story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )


    # ========================================================
    # 3. TECHNIQUE D CONFIGURATION
    # ========================================================

    story.append(
        PageBreak()
    )


    story.append(
        Paragraph(
            "3. Selected Technique D Configuration",
            heading_style,
        )
    )


    configuration_data = [
        [
            "Parameter",
            "Selected Value",
        ],

        [
            "Selected Technique",
            "Technique D",
        ],

        [
            "Processing Method",
            Paragraph(
                (
                    "Bilateral Filtering + Multi-scale "
                    "Top-Hat/Black-Hat Morphological "
                    "Enhancement"
                ),
                body_style,
            ),
        ],

        [
            "Bilateral Diameter",
            str(
                technique_configuration.get(
                    "Bilateral Diameter",
                    3,
                )
            ),
        ],

        [
            "Sigma Colour",
            str(
                technique_configuration.get(
                    "Sigma Colour",
                    15,
                )
            ),
        ],

        [
            "Sigma Space",
            str(
                technique_configuration.get(
                    "Sigma Space",
                    3,
                )
            ),
        ],

        [
            "Morphology Kernel Sizes",
            "7 x 7, 15 x 15, 31 x 31",
        ],

        [
            "Morphology Kernel Shape",
            str(
                technique_configuration.get(
                    "Morphology Kernel Shape",
                    "Ellipse",
                )
            ),
        ],

        [
            "Top-Hat Weight (alpha)",
            str(
                technique_configuration.get(
                    "Alpha",
                    0.7,
                )
            ),
        ],

        [
            "Black-Hat Weight (beta)",
            str(
                technique_configuration.get(
                    "Beta",
                    0.7,
                )
            ),
        ],
    ]


    configuration_table = Table(
        configuration_data,
        colWidths=[
            70 * mm,
            100 * mm,
        ],
    )


    configuration_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        configuration_table
    )


    # ========================================================
    # 4. INDIVIDUAL VISUAL RESULTS
    # ========================================================

    if successful_results:

        story.append(
            PageBreak()
        )


        story.append(
            Paragraph(
                "4. Individual Inspection Visualisations",
                heading_style,
            )
        )


        for index, result in enumerate(
            successful_results,
            start=1,
        ):

            file_name = (
                result.get(
                    "file_name",
                    "Unknown",
                )
            )


            total = int(
                result.get(
                    "total_defects",
                    0,
                )
            )


            confidence = float(
                result.get(
                    "average_confidence",
                    0.0,
                )
            )


            story.append(
                Paragraph(
                    (
                        f"4.{index} {file_name}"
                    ),
                    heading_style,
                )
            )


            detail_data = [
                [
                    "Total Defects",
                    str(
                        total
                    ),
                ],

                [
                    "Most Common Defect",
                    format_defect_name(
                        result.get(
                            "most_common_defect",
                            "None",
                        )
                    ),
                ],

                [
                    "Average Confidence",
                    (
                        f"{confidence:.2%}"
                        if total > 0
                        else "N/A"
                    ),
                ],

                [
                    "Perspective Rectification",
                    (
                        "Applied"
                        if result.get(
                            "rectification_applied",
                            False,
                        )
                        else "Not Applied"
                    ),
                ],

                [
                    "Boundary Method",
                    result.get(
                        "boundary_method",
                        "Not Applied",
                    ),
                ],
            ]


            detail_table = Table(
                detail_data,
                colWidths=[
                    65 * mm,
                    105 * mm,
                ],
            )


            detail_table.setStyle(
                TableStyle(
                    [
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey,
                        ),

                        (
                            "FONTNAME",
                            (0, 0),
                            (0, -1),
                            "Helvetica-Bold",
                        ),

                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            8,
                        ),

                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP",
                        ),

                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            6,
                        ),

                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            4,
                        ),

                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            4,
                        ),
                    ]
                )
            )


            story.append(
                detail_table
            )


            story.append(
                Spacer(
                    1,
                    4 * mm,
                )
            )


            annotated_image = (
                result.get(
                    "annotated_image"
                )
            )


            if annotated_image is not None:

                image_buffer = (
                    encode_image_for_pdf(
                        annotated_image
                    )
                )


                display_width, display_height = (
                    calculate_report_image_size(
                        annotated_image,
                        maximum_width=160 * mm,
                        maximum_height=95 * mm,
                    )
                )


                report_image = Image(
                    image_buffer,
                    width=display_width,
                    height=display_height,
                )


                story.append(
                    report_image
                )


            if index < len(
                successful_results
            ):

                story.append(
                    PageBreak()
                )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


    pdf_buffer.seek(0)


    return (
        pdf_buffer.getvalue()
    )