import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.preprocessing import (
    apply_technique_d,
    get_technique_d_configuration,
)

from modules.detection import (
    detect_defects,
    draw_detections,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Processing Analysis",
    page_icon="🖼️",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_uploaded_image(uploaded_file):
    """
    Convert a Streamlit uploaded image into an
    OpenCV three-channel BGR image.
    """

    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
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


def format_defect_name(name):
    """
    Convert internal class names into readable labels.
    """

    return (
        name
        .replace("_", " ")
        .title()
    )


def create_detection_dataframe(detections):
    """
    Convert YOLO detections into a display table.
    """

    rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        rows.append(
            {
                "No.": index,
                "Defect Class": format_defect_name(
                    detection["class_name"]
                ),
                "Confidence (%)": round(
                    detection["confidence"] * 100,
                    2,
                ),
                "X1": round(
                    detection["x1"],
                    1,
                ),
                "Y1": round(
                    detection["y1"],
                    1,
                ),
                "X2": round(
                    detection["x2"],
                    1,
                ),
                "Y2": round(
                    detection["y2"],
                    1,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Image Processing Analysis"
)

st.caption(
    "Visual analysis of the selected Technique D preprocessing "
    "pipeline before YOLOv8n PCB defect detection."
)

st.divider()


# ============================================================
# IMAGE SOURCE
# ============================================================

st.subheader(
    "1. Select PCB Image"
)


last_inspection_available = (
    "last_inspection" in st.session_state
    and st.session_state.last_inspection is not None
)


source_options = [
    "Upload New Image",
]


if last_inspection_available:
    source_options.insert(
        0,
        "Use Latest Inspection Image",
    )


image_source = st.radio(
    "Image Source",
    source_options,
    horizontal=True,
)


analysis_image = None
analysis_file_name = None


# ============================================================
# USE LATEST INSPECTION IMAGE
# ============================================================

if (
    image_source
    == "Use Latest Inspection Image"
):

    inspection = (
        st.session_state.last_inspection
    )

    analysis_image = (
        inspection["original_image"]
        .copy()
    )

    analysis_file_name = (
        inspection["file_name"]
    )


    st.success(
        f"Using latest inspected image: "
        f"{analysis_file_name}"
    )


# ============================================================
# UPLOAD NEW IMAGE
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Choose a PCB image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ],
        key="processing_analysis_upload",
    )


    if uploaded_file is not None:

        try:

            analysis_image = (
                decode_uploaded_image(
                    uploaded_file
                )
            )

            analysis_file_name = (
                uploaded_file.name
            )

        except ValueError as error:

            st.error(
                str(error)
            )

            st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

if analysis_image is not None:

    image_height, image_width = (
        analysis_image.shape[:2]
    )


    info_col_1, info_col_2 = (
        st.columns(2)
    )


    info_col_1.write(
        f"**File Name:** "
        f"{analysis_file_name}"
    )


    info_col_2.write(
        f"**Dimensions:** "
        f"{image_width} × "
        f"{image_height} pixels"
    )


    # ========================================================
    # APPLY COMPLETE TECHNIQUE D PIPELINE
    # ========================================================

    try:

        with st.spinner(
            "Applying Technique D preprocessing..."
        ):

            stages = (
                apply_technique_d(
                    analysis_image,
                    return_stages=True,
                )
            )


            detections, _ = (
                detect_defects(
                    stages["enhanced"]
                )
            )


            annotated_image = (
                draw_detections(
                    analysis_image,
                    detections,
                )
            )


    except Exception as error:

        st.error(
            f"Processing analysis failed: {error}"
        )

        st.stop()


    st.success(
        "Technique D processing completed successfully."
    )


    st.divider()


    # ========================================================
    # 2. ORIGINAL VS FINAL ENHANCED IMAGE
    # ========================================================

    st.subheader(
        "2. Original and Final Enhanced Image"
    )


    original_col, enhanced_col = (
        st.columns(2)
    )


    with original_col:

        st.image(
            bgr_to_rgb(
                stages["original"]
            ),
            caption="Original PCB Image",
            use_container_width=True,
        )


    with enhanced_col:

        st.image(
            bgr_to_rgb(
                stages["enhanced"]
            ),
            caption="Final Technique D Enhanced Image",
            use_container_width=True,
        )


    st.divider()


    # ========================================================
    # 3. LAB LIGHTNESS PROCESSING
    # ========================================================

    st.subheader(
        "3. Lightness Channel Processing"
    )


    st.write(
        "Technique D converts the original BGR image to LAB "
        "colour space and performs enhancement on the "
        "lightness channel while preserving the original "
        "colour channels."
    )


    lightness_col_1, lightness_col_2, lightness_col_3 = (
        st.columns(3)
    )


    with lightness_col_1:

        st.image(
            stages["original_lightness"],
            caption="Original LAB Lightness",
            use_container_width=True,
            clamp=True,
        )


    with lightness_col_2:

        st.image(
            stages["filtered_lightness"],
            caption="Bilateral-Filtered Lightness",
            use_container_width=True,
            clamp=True,
        )


    with lightness_col_3:

        st.image(
            stages["enhanced_lightness"],
            caption="Enhanced Lightness",
            use_container_width=True,
            clamp=True,
        )


    st.divider()


    # ========================================================
    # 4. MULTI-SCALE MORPHOLOGICAL ENHANCEMENT
    # ========================================================

    st.subheader(
        "4. Multi-Scale Morphological Enhancement"
    )


    st.write(
        "Top-Hat morphology emphasises locally bright "
        "structures, while Black-Hat morphology emphasises "
        "locally dark structures. Technique D evaluates "
        "three different kernel sizes and retains the "
        "strongest response at each pixel."
    )


    configuration = (
        get_technique_d_configuration()
    )


    kernel_sizes = (
        configuration[
            "Morphology Kernels"
        ]
    )


    # ========================================================
    # INDIVIDUAL TOP-HAT MAPS
    # ========================================================

    st.markdown(
        "#### Individual Top-Hat Responses"
    )


    top_hat_columns = (
        st.columns(
            len(kernel_sizes)
        )
    )


    for (
        column,
        kernel_size,
        map_image,
    ) in zip(
        top_hat_columns,
        kernel_sizes,
        stages["white_maps"],
    ):

        with column:

            st.image(
                map_image,
                caption=(
                    f"Top-Hat "
                    f"{kernel_size} × "
                    f"{kernel_size}"
                ),
                use_container_width=True,
                clamp=True,
            )


    # ========================================================
    # INDIVIDUAL BLACK-HAT MAPS
    # ========================================================

    st.markdown(
        "#### Individual Black-Hat Responses"
    )


    black_hat_columns = (
        st.columns(
            len(kernel_sizes)
        )
    )


    for (
        column,
        kernel_size,
        map_image,
    ) in zip(
        black_hat_columns,
        kernel_sizes,
        stages["black_maps"],
    ):

        with column:

            st.image(
                map_image,
                caption=(
                    f"Black-Hat "
                    f"{kernel_size} × "
                    f"{kernel_size}"
                ),
                use_container_width=True,
                clamp=True,
            )


    # ========================================================
    # MAXIMUM FUSION MAPS
    # ========================================================

    st.markdown(
        "#### Maximum-Fused Morphology Maps"
    )


    fusion_col_1, fusion_col_2 = (
        st.columns(2)
    )


    with fusion_col_1:

        st.image(
            stages["white_top_hat"],
            caption=(
                "Maximum-Fused Top-Hat Map"
            ),
            use_container_width=True,
            clamp=True,
        )


    with fusion_col_2:

        st.image(
            stages["black_hat"],
            caption=(
                "Maximum-Fused Black-Hat Map"
            ),
            use_container_width=True,
            clamp=True,
        )


    st.divider()


    # ========================================================
    # 5. FINAL ENHANCEMENT EQUATION
    # ========================================================

    st.subheader(
        "5. Final Enhancement"
    )


    st.markdown(
        """
        The final lightness channel is produced using:

        **Enhanced Lightness = Filtered Lightness
        + α × Top-Hat − β × Black-Hat**
        """
    )


    equation_col_1, equation_col_2, equation_col_3 = (
        st.columns(3)
    )


    equation_col_1.metric(
        "Alpha",
        configuration[
            "Alpha"
        ],
    )


    equation_col_2.metric(
        "Beta",
        configuration[
            "Beta"
        ],
    )


    equation_col_3.metric(
        "Kernel Shape",
        configuration[
            "Morphology Kernel Shape"
        ],
    )


    final_col_1, final_col_2 = (
        st.columns(2)
    )


    with final_col_1:

        st.image(
            stages["enhanced_lightness"],
            caption="Final Enhanced Lightness",
            use_container_width=True,
            clamp=True,
        )


    with final_col_2:

        st.image(
            bgr_to_rgb(
                stages["enhanced"]
            ),
            caption="Reconstructed Technique D Image",
            use_container_width=True,
        )


    st.divider()


    # ========================================================
    # 6. FINAL YOLO DETECTION
    # ========================================================

    st.subheader(
        "6. YOLOv8n Detection after Technique D"
    )


    detection_col_1, detection_col_2 = (
        st.columns(2)
    )


    with detection_col_1:

        st.image(
            bgr_to_rgb(
                stages["enhanced"]
            ),
            caption=(
                "Image Supplied to YOLOv8n"
            ),
            use_container_width=True,
        )


    with detection_col_2:

        st.image(
            bgr_to_rgb(
                annotated_image
            ),
            caption=(
                "Final Defect Detection Result"
            ),
            use_container_width=True,
        )


    if detections:

        st.success(
            f"{len(detections)} defect(s) detected."
        )


        detection_dataframe = (
            create_detection_dataframe(
                detections
            )
        )


        st.dataframe(
            detection_dataframe,
            use_container_width=True,
            hide_index=True,
        )


    else:

        st.info(
            "No defects were detected in this image."
        )


    st.divider()


    # ========================================================
    # 7. FINAL TECHNIQUE D CONFIGURATION
    # ========================================================

    st.subheader(
        "7. Selected Technique D Configuration"
    )


    configuration_rows = []


    for parameter, value in (
        configuration.items()
    ):

        if isinstance(
            value,
            list,
        ):
            value = ", ".join(
                str(item)
                for item in value
            )


        configuration_rows.append(
            {
                "Parameter":
                    parameter,

                "Selected Value":
                    value,
            }
        )


    configuration_dataframe = (
        pd.DataFrame(
            configuration_rows
        )
    )


    st.dataframe(
        configuration_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "These parameters are fixed because they represent "
        "the final Technique D configuration selected during "
        "experimental evaluation."
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload a PCB image or complete an inspection "
        "first to view the Technique D processing stages."
    )