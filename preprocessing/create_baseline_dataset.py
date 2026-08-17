import shutil
from pathlib import Path


# ---------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset"
)

OUTPUT_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_baseline"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ---------------------------------------------------------
# Find corresponding label
# ---------------------------------------------------------

def find_label(image_path, labels_folder):
    """
    Find the YOLO label corresponding to an image.

    Handles:
    1. Normal matching:
       image001.jpg -> image001.txt

    2. Dataset-specific matching:
       xxx_600.jpg -> xxx_256.txt
    """

    # Normal filename matching
    normal_label = labels_folder / f"{image_path.stem}.txt"

    if normal_label.exists():
        return normal_label

    # Special _600 image -> _256 label case
    if image_path.stem.endswith("_600"):

        alternative_stem = (
            image_path.stem[:-4] + "_256"
        )

        alternative_label = (
            labels_folder /
            f"{alternative_stem}.txt"
        )

        if alternative_label.exists():
            return alternative_label

    return None


# ---------------------------------------------------------
# Process one split
# ---------------------------------------------------------

def process_split(split):

    source_images = (
        SOURCE_DATASET / split / "images"
    )

    source_labels = (
        SOURCE_DATASET / split / "labels"
    )

    output_images = (
        OUTPUT_DATASET / split / "images"
    )

    output_labels = (
        OUTPUT_DATASET / split / "labels"
    )

    # Create output folders
    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find image files
    image_files = sorted([
        file
        for file in source_images.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ])

    copied_images = 0
    copied_labels = 0
    missing_labels = 0

    print(f"\nProcessing {split}...")
    print(f"Images found: {len(image_files)}")

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # -------------------------------------------------
        # Copy ORIGINAL image without preprocessing
        # -------------------------------------------------

        output_image_path = (
            output_images /
            image_path.name
        )

        shutil.copy2(
            image_path,
            output_image_path
        )

        copied_images += 1

        # -------------------------------------------------
        # Find corresponding label
        # -------------------------------------------------

        label_path = find_label(
            image_path,
            source_labels
        )

        if label_path is not None:

            # IMPORTANT:
            # Output label must use the SAME filename stem
            # as the image.
            #
            # Example:
            #
            # image:
            # xxx_600.jpg
            #
            # original label:
            # xxx_256.txt
            #
            # baseline label becomes:
            # xxx_600.txt

            output_label_path = (
                output_labels /
                f"{image_path.stem}.txt"
            )

            shutil.copy2(
                label_path,
                output_label_path
            )

            copied_labels += 1

        else:

            print(
                f"WARNING: Label not found for "
                f"{image_path.name}"
            )

            missing_labels += 1

        # Progress update
        if index % 500 == 0:

            print(
                f"{split}: "
                f"{index}/{len(image_files)} processed"
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(f"\n{split} completed")
    print(f"Images copied: {copied_images}")
    print(f"Labels copied: {copied_labels}")
    print(f"Missing labels: {missing_labels}")


# ---------------------------------------------------------
# Create YOLO data.yaml
# ---------------------------------------------------------

def create_data_yaml():

    yaml_path = (
        OUTPUT_DATASET /
        "data.yaml"
    )

    dataset_path = (
        OUTPUT_DATASET
        .resolve()
        .as_posix()
    )

    yaml_content = f"""path: "{dataset_path}"

train: train/images
val: val/images
test: test/images

names:
  0: mouse_bite
  1: spur
  2: missing_hole
  3: short
  4: open_circuit
  5: spurious_copper
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8"
    )

    print(
        f"\nCreated: {yaml_path}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Creating baseline PCB dataset..."
    )

    print(
        "No image preprocessing will be applied."
    )

    for split in SPLITS:
        process_split(split)

    create_data_yaml()

    print(
        "\nBaseline dataset creation completed."
    )