import cv2
import shutil
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================

CLIP_LIMITS = [1.0, 2.0, 3.0]
TILE_GRID_SIZE = (8, 8)

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# =========================================================
# CLAHE
# =========================================================

def apply_clahe(image, clip_limit, tile_grid_size=(8, 8)):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# =========================================================
# FIND LABEL
# =========================================================

def find_label(image_path, labels_folder):

    # Normal case
    normal_label = (
        labels_folder /
        f"{image_path.stem}.txt"
    )

    if normal_label.exists():
        return normal_label

    # Dataset-specific case:
    # xxx_600.jpg → xxx_256.txt
    if image_path.stem.endswith("_600"):

        alternative_stem = (
            image_path.stem[:-4]
            + "_256"
        )

        alternative_label = (
            labels_folder /
            f"{alternative_stem}.txt"
        )

        if alternative_label.exists():
            return alternative_label

    return None


# =========================================================
# PROCESS ONE SPLIT
# =========================================================

def process_split(
    split,
    clip_limit,
    output_dataset
):

    source_images = (
        SOURCE_DATASET /
        split /
        "images"
    )

    source_labels = (
        SOURCE_DATASET /
        split /
        "labels"
    )

    output_images = (
        output_dataset /
        split /
        "images"
    )

    output_labels = (
        output_dataset /
        split /
        "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted([
        file
        for file in source_images.iterdir()
        if file.suffix.lower()
        in IMAGE_EXTENSIONS
    ])

    processed_count = 0
    copied_labels = 0
    missing_labels = 0

    print(f"\nProcessing {split}")
    print(f"Clip Limit: {clip_limit}")
    print(f"Images found: {len(image_files)}")

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: Cannot read "
                f"{image_path.name}"
            )
            continue

        # Apply current CLAHE setting
        enhanced_image = apply_clahe(
            image,
            clip_limit=clip_limit,
            tile_grid_size=TILE_GRID_SIZE
        )

        output_image_path = (
            output_images /
            image_path.name
        )

        success = cv2.imwrite(
            str(output_image_path),
            enhanced_image
        )

        if not success:
            print(
                f"WARNING: Cannot save "
                f"{image_path.name}"
            )
            continue

        processed_count += 1

        # Find label
        label_path = find_label(
            image_path,
            source_labels
        )

        if label_path is not None:

            # Make output label name match image name
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

            missing_labels += 1

            print(
                f"WARNING: Label not found for "
                f"{image_path.name}"
            )

        if index % 500 == 0:

            print(
                f"{index}/{len(image_files)} "
                f"processed"
            )

    print(f"\n{split} completed")
    print(f"Images processed: {processed_count}")
    print(f"Labels copied: {copied_labels}")
    print(f"Missing labels: {missing_labels}")


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_data_yaml(output_dataset):

    yaml_path = (
        output_dataset /
        "data.yaml"
    )

    dataset_path = (
        output_dataset
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


# =========================================================
# PROCESS ONE COMPLETE CLAHE CONFIGURATION
# =========================================================

def process_clahe_configuration(clip_limit):

    output_dataset = Path(
        rf"C:\Users\tiong\Downloads"
        rf"\pcb_dataset_clahe_{clip_limit}"
    )

    print("\n" + "=" * 60)
    print(
        f"STARTING CLAHE "
        f"clipLimit = {clip_limit}"
    )
    print("=" * 60)

    for split in SPLITS:

        process_split(
            split=split,
            clip_limit=clip_limit,
            output_dataset=output_dataset
        )

    create_data_yaml(
        output_dataset
    )

    print("\nCompleted:")
    print(output_dataset)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CLAHE PARAMETER DATASET GENERATOR")
    print("=" * 60)

    print(
        f"Testing Clip Limits: "
        f"{CLIP_LIMITS}"
    )

    print(
        f"Tile Grid Size: "
        f"{TILE_GRID_SIZE}"
    )

    # Automatically run all three
    for clip_limit in CLIP_LIMITS:

        process_clahe_configuration(
            clip_limit
        )

    print("\n" + "=" * 60)
    print("ALL CLAHE DATASETS COMPLETED")
    print("=" * 60)