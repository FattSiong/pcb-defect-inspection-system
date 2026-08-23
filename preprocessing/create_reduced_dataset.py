from pathlib import Path
import random
import shutil
import csv


# =========================================================
# SETTINGS
# =========================================================

SOURCE_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset"
)

OUTPUT_DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

IMAGES_PER_CLASS = 200

# Fixed random seed so all group members
# can reproduce the same selection
SEED = 42


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# =========================================================
# FIND LABEL
# =========================================================

def find_label(image_path, labels_folder):
    """
    Find the YOLO label for an image.

    Handles the dataset mismatch:

    image:
        xxx_600.jpg

    label:
        xxx_256.txt
    """

    # Normal case
    normal_label = (
        labels_folder /
        f"{image_path.stem}.txt"
    )

    if normal_label.exists():
        return normal_label


    # Dataset-specific mismatch
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
# READ CLASSES FROM LABEL
# =========================================================

def read_classes(label_path):

    classes = set()

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            classes.add(class_id)

    return classes


# =========================================================
# GET IMAGE FILES
# =========================================================

def get_images(folder):

    return sorted([
        file
        for file in folder.iterdir()
        if file.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


# =========================================================
# CREATE REDUCED TRAIN SET
# =========================================================

def create_reduced_train():

    print("\n" + "=" * 70)
    print("CREATING REDUCED TRAIN DATASET")
    print("=" * 70)


    source_images = (
        SOURCE_DATASET /
        "train" /
        "images"
    )

    source_labels = (
        SOURCE_DATASET /
        "train" /
        "labels"
    )


    output_images = (
        OUTPUT_DATASET /
        "train" /
        "images"
    )

    output_labels = (
        OUTPUT_DATASET /
        "train" /
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


    image_files = get_images(
        source_images
    )


    print(
        f"Original training images: "
        f"{len(image_files)}"
    )


    # =====================================================
    # GROUP IMAGES BY DEFECT CLASS
    # =====================================================

    class_images = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    image_to_label = {}

    skipped_multi_class = 0
    missing_labels = 0


    for image_path in image_files:

        label_path = find_label(
            image_path,
            source_labels
        )


        if label_path is None:

            missing_labels += 1

            continue


        classes = read_classes(
            label_path
        )


        if len(classes) == 0:
            continue


        # -------------------------------------------------
        # Only use images containing ONE defect class.
        #
        # Multiple instances of the SAME class are allowed.
        #
        # Example:
        # 2 ...
        # 2 ...
        #
        # This is still a missing_hole image.
        # -------------------------------------------------

        if len(classes) > 1:

            skipped_multi_class += 1

            continue


        class_id = next(
            iter(classes)
        )


        if class_id not in CLASS_NAMES:
            continue


        class_images[
            class_id
        ].append(
            image_path
        )


        image_to_label[
            image_path
        ] = label_path


    # =====================================================
    # SHOW AVAILABLE IMAGES
    # =====================================================

    print("\nAvailable images per class:")

    for class_id, class_name in CLASS_NAMES.items():

        available = len(
            class_images[class_id]
        )

        print(
            f"{class_id} "
            f"{class_name:<20}: "
            f"{available}"
        )


    print(
        f"\nSkipped multi-class images: "
        f"{skipped_multi_class}"
    )

    print(
        f"Missing labels: "
        f"{missing_labels}"
    )


    # =====================================================
    # CHECK FOR AT LEAST 200 PER CLASS
    # =====================================================

    for class_id, class_name in CLASS_NAMES.items():

        available = len(
            class_images[class_id]
        )

        if available < IMAGES_PER_CLASS:

            raise ValueError(
                f"\nNot enough images for "
                f"{class_name}.\n"
                f"Required: {IMAGES_PER_CLASS}\n"
                f"Available: {available}"
            )


    # =====================================================
    # SELECT EXACTLY 200 PER CLASS
    # =====================================================

    rng = random.Random(
        SEED
    )

    selected_data = []


    print("\nSelecting 200 images per defect:")


    for class_id, class_name in CLASS_NAMES.items():

        candidates = sorted(
            class_images[class_id]
        )


        selected = rng.sample(
            candidates,
            IMAGES_PER_CLASS
        )


        print(
            f"{class_name:<20}: "
            f"{len(selected)}"
        )


        for image_path in selected:

            selected_data.append({
                "class_id": class_id,
                "class_name": class_name,
                "image": image_path,
                "label": image_to_label[
                    image_path
                ]
            })


    # =====================================================
    # COPY SELECTED TRAINING DATA
    # =====================================================

    print("\nCopying selected training data...")


    for index, item in enumerate(
        selected_data,
        start=1
    ):

        image_path = item[
            "image"
        ]

        label_path = item[
            "label"
        ]


        # Copy image
        shutil.copy2(
            image_path,
            output_images /
            image_path.name
        )


        # IMPORTANT:
        # output label uses same stem as image
        #
        # xxx_600.jpg
        # ->
        # xxx_600.txt

        output_label_path = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        shutil.copy2(
            label_path,
            output_label_path
        )


        if index % 200 == 0:

            print(
                f"{index}/"
                f"{len(selected_data)} copied"
            )


    # =====================================================
    # SAVE SELECTION MANIFEST
    # =====================================================

    manifest_path = (
        OUTPUT_DATASET /
        "selected_train_images.csv"
    )


    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "image_name",
            "source_label"
        ])


        for item in selected_data:

            writer.writerow([
                item["class_id"],
                item["class_name"],
                item["image"].name,
                item["label"].name
            ])


    print(
        f"\nTraining subset completed."
    )

    print(
        f"Total training images: "
        f"{len(selected_data)}"
    )


# =========================================================
# COPY FULL VAL / TEST
# =========================================================

def copy_full_split(split):

    print("\n" + "=" * 70)
    print(
        f"COPYING FULL {split.upper()} DATASET"
    )
    print("=" * 70)


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
        OUTPUT_DATASET /
        split /
        "images"
    )

    output_labels = (
        OUTPUT_DATASET /
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


    image_files = get_images(
        source_images
    )


    copied_images = 0
    copied_labels = 0
    missing_labels = 0


    print(
        f"{len(image_files)} images found"
    )


    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # Copy image
        shutil.copy2(
            image_path,
            output_images /
            image_path.name
        )

        copied_images += 1


        # Find corresponding label
        label_path = find_label(
            image_path,
            source_labels
        )


        if label_path is None:

            print(
                f"WARNING: label not found for "
                f"{image_path.name}"
            )

            missing_labels += 1

            continue


        # Fix label filename if needed
        output_label_path = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        shutil.copy2(
            label_path,
            output_label_path
        )

        copied_labels += 1


        if index % 500 == 0:

            print(
                f"{index}/"
                f"{len(image_files)} copied"
            )


    print(
        f"\n{split.upper()} completed"
    )

    print(
        f"Images: {copied_images}"
    )

    print(
        f"Labels: {copied_labels}"
    )

    print(
        f"Missing labels: "
        f"{missing_labels}"
    )


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_data_yaml():

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


    yaml_path = (
        OUTPUT_DATASET /
        "data.yaml"
    )


    yaml_path.write_text(
        yaml_content,
        encoding="utf-8"
    )


    print(
        f"\ndata.yaml created:\n"
        f"{yaml_path}"
    )


# =========================================================
# VERIFY DATASET
# =========================================================

def verify_dataset():

    print("\n" + "=" * 70)
    print("FINAL DATASET VERIFICATION")
    print("=" * 70)


    for split in [
        "train",
        "val",
        "test"
    ]:

        image_folder = (
            OUTPUT_DATASET /
            split /
            "images"
        )

        label_folder = (
            OUTPUT_DATASET /
            split /
            "labels"
        )


        images = get_images(
            image_folder
        )

        labels = list(
            label_folder.glob(
                "*.txt"
            )
        )


        print(
            f"{split:<6} "
            f"Images: {len(images):<6} "
            f"Labels: {len(labels):<6}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("PCB REDUCED DATASET GENERATOR")
    print("=" * 70)


    print(
        f"Source:\n"
        f"{SOURCE_DATASET}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DATASET}"
    )

    print(
        f"\nTraining images per class: "
        f"{IMAGES_PER_CLASS}"
    )

    print(
        f"Random seed: "
        f"{SEED}"
    )


    # Reduced TRAIN
    create_reduced_train()


    # Full VAL
    copy_full_split(
        "val"
    )


    # Full TEST
    copy_full_split(
        "test"
    )


    # data.yaml
    create_data_yaml()


    # Verification
    verify_dataset()


    print("\n" + "=" * 70)
    print("REDUCED DATASET COMPLETED")
    print("=" * 70)


    print(
        f"\nNew dataset:\n"
        f"{OUTPUT_DATASET}"
    )


if __name__ == "__main__":
    main()