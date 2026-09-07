from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    MODEL_IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    DEFECT_CLASSES,
)


# ============================================================
# LOAD FINAL TECHNIQUE D YOLO MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_detection_model():
    """
    Load the final YOLOv8n model trained using
    Technique D preprocessed PCB images.

    The model is cached so that it is loaded only once
    during the application's execution.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Technique D model was not found:\n{MODEL_PATH}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    return model


# ============================================================
# RUN DEFECT DETECTION
# ============================================================

def detect_defects(
    processed_image: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
):
    """
    Run the final Technique D YOLO model on a
    preprocessed PCB image.

    Parameters
    ----------
    processed_image : np.ndarray
        Technique D enhanced PCB image in BGR format.

    confidence_threshold : float
        Minimum confidence threshold for detections.

    Returns
    -------
    tuple
        detections : list[dict]
            Structured detection information.

        raw_result
            Original Ultralytics result object.
    """

    if processed_image is None:
        raise ValueError(
            "Processed image cannot be None."
        )

    if not isinstance(
        processed_image,
        np.ndarray,
    ):
        raise ValueError(
            "Processed image must be a NumPy array."
        )

    if (
        processed_image.ndim != 3
        or processed_image.shape[2] != 3
    ):
        raise ValueError(
            "Processed image must be a "
            "three-channel BGR image."
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_detection_model()


    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model.predict(
        source=processed_image,
        imgsz=MODEL_IMAGE_SIZE,
        conf=confidence_threshold,
        verbose=False,
    )


    if not results:
        return [], None


    result = results[0]

    detections = []


    # --------------------------------------------------------
    # Extract bounding boxes
    # --------------------------------------------------------

    if result.boxes is None:
        return detections, result


    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )


        detection = {
            "class_id":
                class_id,

            "class_name":
                DEFECT_CLASSES.get(
                    class_id,
                    f"class_{class_id}",
                ),

            "confidence":
                confidence,

            "x1":
                float(x1),

            "y1":
                float(y1),

            "x2":
                float(x2),

            "y2":
                float(y2),

            "width":
                float(x2 - x1),

            "height":
                float(y2 - y1),
        }


        detections.append(
            detection
        )


    return detections, result


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image: np.ndarray,
    detections,
) -> np.ndarray:
    """
    Draw defect bounding boxes and labels onto an image.

    Because Technique D does not modify image geometry,
    the bounding boxes predicted from the processed image
    can also be drawn directly onto the original PCB image.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    annotated_image = image.copy()


    for detection in detections:

        x1 = int(
            detection["x1"]
        )

        y1 = int(
            detection["y1"]
        )

        x2 = int(
            detection["x2"]
        )

        y2 = int(
            detection["y2"]
        )

        class_name = (
            detection["class_name"]
        )

        confidence = (
            detection["confidence"]
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )


        # ----------------------------------------------------
        # Detection label
        # ----------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        text_width, text_height = (
            text_size
        )


        label_y1 = max(
            y1 - text_height - 10,
            0,
        )


        cv2.rectangle(
            annotated_image,
            (
                x1,
                label_y1,
            ),
            (
                x1 + text_width + 6,
                y1,
            ),
            (0, 255, 0),
            -1,
        )


        cv2.putText(
            annotated_image,
            label,
            (
                x1 + 3,
                y1 - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


    return annotated_image


# ============================================================
# DEFECT SUMMARY
# ============================================================

def summarise_detections(
    detections
):
    """
    Produce a simple summary of detected PCB defects.
    """

    summary = {
        class_name: 0
        for class_name
        in DEFECT_CLASSES.values()
    }


    for detection in detections:

        class_name = (
            detection["class_name"]
        )

        if class_name in summary:
            summary[class_name] += 1


    return summary