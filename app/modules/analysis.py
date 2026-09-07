import pandas as pd


# ============================================================
# DEFECT NAME FORMATTING
# ============================================================

def format_defect_name(defect_name):
    """
    Convert internal defect class names into readable labels.

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


# ============================================================
# TOTAL IMAGES INSPECTED
# ============================================================

def calculate_total_images(history):
    """
    Return the total number of PCB images inspected.
    """

    if not history:
        return 0

    return len(history)


# ============================================================
# TOTAL DEFECTS
# ============================================================

def calculate_total_defects(history):
    """
    Calculate the total number of detected defects
    across all inspected PCB images.
    """

    if not history:
        return 0

    return sum(
        int(
            record.get(
                "Total Defects",
                0,
            )
        )
        for record in history
    )


# ============================================================
# AVERAGE CONFIDENCE
# ============================================================

def calculate_overall_average_confidence(history):
    """
    Calculate the average confidence across inspections
    that contain at least one detected defect.

    Inspection records with no detections are excluded
    because their confidence value is not applicable.
    """

    if not history:
        return 0.0

    valid_confidences = []

    for record in history:

        total_defects = int(
            record.get(
                "Total Defects",
                0,
            )
        )

        confidence = float(
            record.get(
                "Average Confidence",
                0.0,
            )
        )

        if total_defects > 0:
            valid_confidences.append(
                confidence
            )

    if not valid_confidences:
        return 0.0

    return sum(
        valid_confidences
    ) / len(
        valid_confidences
    )


# ============================================================
# AGGREGATED DEFECT DISTRIBUTION
# ============================================================

def aggregate_defect_distribution(history):
    """
    Aggregate all six PCB defect classes across
    inspection-history records.

    Requires each history record to contain:
        "Defect Summary"
    """

    distribution = {
        "mouse_bite": 0,
        "spur": 0,
        "missing_hole": 0,
        "short": 0,
        "open_circuit": 0,
        "spurious_copper": 0,
    }

    if not history:
        return distribution

    for record in history:

        defect_summary = record.get(
            "Defect Summary",
            {},
        )

        if not isinstance(
            defect_summary,
            dict,
        ):
            continue

        for (
            defect_class,
            count,
        ) in defect_summary.items():

            if defect_class in distribution:

                distribution[
                    defect_class
                ] += int(
                    count
                )

    return distribution


# ============================================================
# MOST COMMON DEFECT
# ============================================================

def find_overall_most_common_defect(history):
    """
    Determine the most frequently detected defect
    across all inspections.
    """

    distribution = (
        aggregate_defect_distribution(
            history
        )
    )

    detected_distribution = {
        defect_class: count
        for defect_class, count
        in distribution.items()
        if count > 0
    }

    if not detected_distribution:
        return "None"

    return max(
        detected_distribution,
        key=detected_distribution.get,
    )


# ============================================================
# INSPECTION STATUS DISTRIBUTION
# ============================================================

def calculate_status_distribution(history):
    """
    Count inspections with detected defects and
    inspections with no detected defects.
    """

    status_distribution = {
        "Defects Detected": 0,
        "No Defects Detected": 0,
    }

    if not history:
        return status_distribution

    for record in history:

        total_defects = int(
            record.get(
                "Total Defects",
                0,
            )
        )

        if total_defects > 0:

            status_distribution[
                "Defects Detected"
            ] += 1

        else:

            status_distribution[
                "No Defects Detected"
            ] += 1

    return status_distribution


# ============================================================
# CALIBRATION USAGE
# ============================================================

def calculate_calibration_usage(history):
    """
    Calculate how often perspective rectification and
    spatial calibration were used.
    """

    usage = {
        "Perspective Rectification": 0,
        "Spatial Calibration": 0,
    }

    if not history:
        return usage

    for record in history:

        if record.get(
            "Rectification Applied",
            False,
        ):

            usage[
                "Perspective Rectification"
            ] += 1


        if record.get(
            "Spatial Calibration",
            False,
        ):

            usage[
                "Spatial Calibration"
            ] += 1

    return usage


# ============================================================
# DEFECT DISTRIBUTION DATAFRAME
# ============================================================

def create_defect_distribution_dataframe(
    history,
):
    """
    Create a DataFrame containing aggregate defect counts.
    """

    distribution = (
        aggregate_defect_distribution(
            history
        )
    )

    rows = []

    for (
        defect_class,
        count,
    ) in distribution.items():

        rows.append(
            {
                "Defect Class":
                    format_defect_name(
                        defect_class
                    ),

                "Detected Count":
                    int(count),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STATUS DATAFRAME
# ============================================================

def create_status_dataframe(history):
    """
    Create a DataFrame containing inspection statuses.
    """

    distribution = (
        calculate_status_distribution(
            history
        )
    )

    return pd.DataFrame(
        [
            {
                "Inspection Status":
                    status,

                "Number of Images":
                    count,
            }

            for status, count
            in distribution.items()
        ]
    )


# ============================================================
# CALIBRATION DATAFRAME
# ============================================================

def create_calibration_dataframe(history):
    """
    Create a DataFrame containing calibration usage.
    """

    usage = (
        calculate_calibration_usage(
            history
        )
    )

    return pd.DataFrame(
        [
            {
                "Calibration Feature":
                    feature,

                "Number of Uses":
                    count,
            }

            for feature, count
            in usage.items()
        ]
    )


# ============================================================
# INSPECTION HISTORY DATAFRAME
# ============================================================

def create_history_dataframe(history):
    """
    Convert Streamlit inspection-history records into
    a clean table for dashboard display.
    """

    if not history:

        return pd.DataFrame(
            columns=[
                "No.",
                "File Name",
                "Total Defects",
                "Most Common Defect",
                "Average Confidence (%)",
                "Status",
                "Rectification",
                "Spatial Calibration",
            ]
        )


    rows = []


    for (
        index,
        record,
    ) in enumerate(
        history,
        start=1,
    ):

        total_defects = int(
            record.get(
                "Total Defects",
                0,
            )
        )


        average_confidence = float(
            record.get(
                "Average Confidence",
                0.0,
            )
        )


        status = (
            "Defects Detected"
            if total_defects > 0
            else "No Defects Detected"
        )


        rows.append(
            {
                "No.":
                    index,

                "File Name":
                    record.get(
                        "File Name",
                        "Unknown",
                    ),

                "Total Defects":
                    total_defects,

                "Most Common Defect":
                    record.get(
                        "Most Common Defect",
                        "None",
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

                "Status":
                    status,

                "Rectification":
                    (
                        "Yes"
                        if record.get(
                            "Rectification Applied",
                            False,
                        )
                        else "No"
                    ),

                "Spatial Calibration":
                    (
                        "Yes"
                        if record.get(
                            "Spatial Calibration",
                            False,
                        )
                        else "No"
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPLETE DASHBOARD SUMMARY
# ============================================================

def create_dashboard_summary(history):
    """
    Produce the main KPI values required by the
    analysis dashboard.
    """

    return {
        "total_images":
            calculate_total_images(
                history
            ),

        "total_defects":
            calculate_total_defects(
                history
            ),

        "most_common_defect":
            find_overall_most_common_defect(
                history
            ),

        "average_confidence":
            calculate_overall_average_confidence(
                history
            ),
    }