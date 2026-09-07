import pandas as pd
import streamlit as st

from modules.analysis import (
    create_dashboard_summary,
    create_defect_distribution_dataframe,
    create_status_dataframe,
    create_calibration_dataframe,
    create_history_dataframe,
    format_defect_name,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []


history = (
    st.session_state.inspection_history
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Analysis Dashboard"
)

st.caption(
    "Summary of PCB inspection results produced "
    "during the current application session."
)

st.divider()


# ============================================================
# EMPTY DASHBOARD
# ============================================================

if not history:

    st.info(
        "No PCB inspection records are available yet. "
        "Complete one or more inspections first."
    )

    st.markdown(
        """
        The dashboard will display:

        - Total PCB images inspected
        - Total defects detected
        - Most common defect class
        - Average detection confidence
        - Defect-class distribution
        - Inspection-status distribution
        - Calibration usage
        - Detailed inspection history
        """
    )

    st.stop()


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

summary = (
    create_dashboard_summary(
        history
    )
)


# ============================================================
# 1. KPI CARDS
# ============================================================

st.subheader(
    "1. Inspection Overview"
)


metric_1, metric_2, metric_3, metric_4 = (
    st.columns(4)
)


metric_1.metric(
    "Images Inspected",
    summary[
        "total_images"
    ],
)


metric_2.metric(
    "Total Defects",
    summary[
        "total_defects"
    ],
)


metric_3.metric(
    "Most Common Defect",
    format_defect_name(
        summary[
            "most_common_defect"
        ]
    ),
)


if summary["total_defects"] > 0:

    confidence_display = (
        f"{summary['average_confidence']:.2%}"
    )

else:

    confidence_display = "N/A"


metric_4.metric(
    "Average Confidence",
    confidence_display,
)


st.divider()


# ============================================================
# 2. DEFECT DISTRIBUTION
# ============================================================

st.subheader(
    "2. Defect Distribution"
)


defect_dataframe = (
    create_defect_distribution_dataframe(
        history
    )
)


st.dataframe(
    defect_dataframe,
    use_container_width=True,
    hide_index=True,
)


detected_defects = (
    defect_dataframe[
        defect_dataframe[
            "Detected Count"
        ]
        > 0
    ]
)


if not detected_defects.empty:

    defect_chart_data = (
        detected_defects
        .set_index(
            "Defect Class"
        )[
            "Detected Count"
        ]
    )


    st.bar_chart(
        defect_chart_data
    )


else:

    st.info(
        "No PCB defects have been detected "
        "during the current session."
    )


st.divider()


# ============================================================
# 3. INSPECTION STATUS DISTRIBUTION
# ============================================================

st.subheader(
    "3. Inspection Status Distribution"
)


status_dataframe = (
    create_status_dataframe(
        history
    )
)


status_col_1, status_col_2 = (
    st.columns(2)
)


with status_col_1:

    st.dataframe(
        status_dataframe,
        use_container_width=True,
        hide_index=True,
    )


with status_col_2:

    status_chart_data = (
        status_dataframe
        .set_index(
            "Inspection Status"
        )[
            "Number of Images"
        ]
    )


    st.bar_chart(
        status_chart_data
    )


st.caption(
    "'No Defects Detected' indicates that the "
    "YOLOv8n model did not produce a defect detection "
    "above the configured confidence threshold. "
    "It should not be interpreted as absolute proof "
    "that the physical PCB is defect-free."
)


st.divider()


# ============================================================
# 4. CALIBRATION USAGE
# ============================================================

st.subheader(
    "4. Calibration Usage"
)


calibration_dataframe = (
    create_calibration_dataframe(
        history
    )
)


calibration_col_1, calibration_col_2 = (
    st.columns(2)
)


with calibration_col_1:

    st.dataframe(
        calibration_dataframe,
        use_container_width=True,
        hide_index=True,
    )


with calibration_col_2:

    calibration_chart_data = (
        calibration_dataframe
        .set_index(
            "Calibration Feature"
        )[
            "Number of Uses"
        ]
    )


    st.bar_chart(
        calibration_chart_data
    )


st.divider()


# ============================================================
# 5. INSPECTION HISTORY
# ============================================================

st.subheader(
    "5. Inspection History"
)


history_dataframe = (
    create_history_dataframe(
        history
    )
)


st.dataframe(
    history_dataframe,
    use_container_width=True,
    hide_index=True,
)


st.write(
    f"**Total inspection records:** "
    f"{len(history_dataframe)}"
)


# ============================================================
# CSV EXPORT
# ============================================================

csv_data = (
    history_dataframe
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)


st.download_button(
    label="Download Inspection History (CSV)",
    data=csv_data,
    file_name=(
        "pcb_inspection_history.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)


st.divider()


# ============================================================
# 6. SESSION MANAGEMENT
# ============================================================

st.subheader(
    "6. Session Management"
)


st.warning(
    "Inspection history is currently stored only "
    "for the active Streamlit session. Restarting "
    "the application will clear these records."
)


clear_history = st.button(
    "Clear Inspection History",
    type="secondary",
    use_container_width=True,
)


if clear_history:

    st.session_state.inspection_history = []

    st.session_state.last_inspection = None

    st.success(
        "Inspection history cleared."
    )

    st.rerun()