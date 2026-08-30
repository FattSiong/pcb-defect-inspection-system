from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from bilateral_morphology import enhance_pcb


SAMPLE_IMAGES = {
    "mouse_bite": (
        0,
        "l_light_01_mouse_bite_01_2_600.jpg",
    ),
    "spur": (
        1,
        "l_light_01_spur_05_2_600.jpg",
    ),
    "missing_hole": (
        2,
        "l_light_01_missing_hole_01_1_600.jpg",
    ),
    "short": (
        3,
        "l_light_01_short_11_4_600.jpg",
    ),
    "open_circuit": (
        4,
        "l_light_01_open_circuit_03_1_600.jpg",
    ),
    "spurious_copper": (
        5,
        "l_light_01_spurious_copper_03_1_600.jpg",
    ),
}


def read_target_boxes(label_path, target_class):
    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}"
        )

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id != target_class:
            continue

        boxes.append(
            tuple(float(value) for value in parts[1:])
        )

    return boxes


def draw_boxes(image, boxes, class_name):
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for x_center, y_center, width, height in boxes:
        box_width = width * image_width
        box_height = height * image_height

        x1 = int(x_center * image_width - box_width / 2)
        y1 = int(y_center * image_height - box_height / 2)
        x2 = int(x_center * image_width + box_width / 2)
        y2 = int(y_center * image_height + box_height / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width - 1, x2)
        y2 = min(image_height - 1, y2)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            output,
            class_name,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def add_title(image, title):
    title_height = 50

    canvas = np.full(
        (
            image.shape[0] + title_height,
            image.shape[1],
            3,
        ),
        255,
        dtype=np.uint8,
    )

    canvas[title_height:] = image

    cv2.putText(
        canvas,
        title,
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def main():
    parser = ArgumentParser(
        description="Preview enhancement for all six PCB defect classes."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Root of the PCB dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/bilateral_morphology_preview/all_classes"
        ),
    )

    args = parser.parse_args()

    image_directory = args.dataset / "train" / "images"
    label_directory = args.dataset / "train" / "labels"

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name, (
        class_id,
        filename,
    ) in SAMPLE_IMAGES.items():
        image_path = image_directory / filename
        label_path = label_directory / (
            image_path.stem + ".txt"
        )

        original = cv2.imread(str(image_path))

        if original is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        enhanced = enhance_pcb(original)

        assert enhanced.shape == original.shape
        assert enhanced.dtype == original.dtype

        boxes = read_target_boxes(
            label_path,
            class_id,
        )

        if not boxes:
            raise ValueError(
                f"No {class_name} label found in {label_path}"
            )

        original_labelled = draw_boxes(
            original,
            boxes,
            class_name,
        )

        enhanced_labelled = draw_boxes(
            enhanced,
            boxes,
            class_name,
        )

        original_panel = add_title(
            original_labelled,
            "Original",
        )

        enhanced_panel = add_title(
            enhanced_labelled,
            "Bilateral + Multi-scale Morphology",
        )

        comparison = cv2.hconcat(
            [original_panel, enhanced_panel]
        )

        output_path = (
            args.output /
            f"{class_name}_comparison.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            comparison,
        )

        if not success:
            raise RuntimeError(
                f"Could not save: {output_path}"
            )

        print(f"Created: {output_path}")

    print("\nAll six previews completed.")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()