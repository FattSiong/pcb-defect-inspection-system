import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Inspection System",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "PCB Defect Inspection System"
)

st.caption(
    "Image-based PCB defect inspection using "
    "Technique D preprocessing and YOLOv8n."
)

st.divider()


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.subheader(
    "System Overview"
)

st.write(
    """
    This prototype performs automated inspection of PCB images
    using the selected Technique D image-processing approach
    together with a trained YOLOv8n object detection model.
    """
)


# ============================================================
# PROCESSING WORKFLOW
# ============================================================

st.subheader(
    "Inspection Workflow"
)

st.markdown(
    """
    **1. Upload PCB Image**

    ↓

    **2. Image Calibration and Validation**

    ↓

    **3. Technique D Preprocessing**

    - Bilateral Filtering
    - Multi-scale Top-Hat Morphology
    - Multi-scale Black-Hat Morphology

    ↓

    **4. YOLOv8n Defect Detection**

    ↓

    **5. Inspection Results and Analysis**
    """
)


# ============================================================
# SUPPORTED DEFECTS
# ============================================================

st.subheader(
    "Supported PCB Defect Classes"
)

defect_classes = [
    "Mouse Bite",
    "Spur",
    "Missing Hole",
    "Short",
    "Open Circuit",
    "Spurious Copper",
]

column_1, column_2 = st.columns(2)

with column_1:

    for defect in defect_classes[:3]:
        st.write(
            f"• {defect}"
        )

with column_2:

    for defect in defect_classes[3:]:
        st.write(
            f"• {defect}"
        )


st.divider()


# ============================================================
# NAVIGATION MESSAGE
# ============================================================

st.info(
    "Select **Inspection** from the sidebar "
    "to begin inspecting a PCB image."
)