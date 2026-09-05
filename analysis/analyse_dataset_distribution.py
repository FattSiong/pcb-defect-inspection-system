from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced"
)

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

SPLITS = ["train", "val", "test"]


def count_dataset():

    # Number of images containing each class
    image_counts = {
        split: defaultdict(int)
        for split in SPLITS
    }

    # Number of bounding-box annotations for each class
    instance_counts = {
        split: defaultdict(int)
        for split in SPLITS
    }

    total_images = {}

    for split in SPLITS:

        image_dir = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"

        image_files = (
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.bmp"))
        )

        total_images[split] = len(image_files)

        for image_path in image_files:

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            classes_in_image = set()

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    class_id = int(parts[0])

                    # Count each bounding box
                    instance_counts[
                        split
                    ][class_id] += 1

                    # Remember class appears in this image
                    classes_in_image.add(
                        class_id
                    )

            # Count image once per class
            for class_id in classes_in_image:

                image_counts[
                    split
                ][class_id] += 1


    # =====================================================
    # DISPLAY IMAGE COUNTS
    # =====================================================

    print("\n")
    print("=" * 80)
    print("IMAGE COUNT PER DEFECT CLASS")
    print("=" * 80)

    header = (
        f"{'Defect Class':<20}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print(header)
    print("-" * 80)

    grand_class_total = 0

    for class_id, class_name in CLASS_NAMES.items():

        train = image_counts["train"][class_id]
        val = image_counts["val"][class_id]
        test = image_counts["test"][class_id]

        total = train + val + test

        grand_class_total += total

        print(
            f"{class_name:<20}"
            f"{train:>10}"
            f"{val:>10}"
            f"{test:>10}"
            f"{total:>10}"
        )


    print("\nActual number of image files:")

    for split in SPLITS:

        print(
            f"{split.capitalize():<10}: "
            f"{total_images[split]}"
        )

    print(
        f"{'Overall':<10}: "
        f"{sum(total_images.values())}"
    )


    # =====================================================
    # DISPLAY ANNOTATION COUNTS
    # =====================================================

    print("\n")
    print("=" * 80)
    print("ANNOTATED DEFECT INSTANCES")
    print("=" * 80)

    print(header)
    print("-" * 80)

    for class_id, class_name in CLASS_NAMES.items():

        train = (
            instance_counts[
                "train"
            ][class_id]
        )

        val = (
            instance_counts[
                "val"
            ][class_id]
        )

        test = (
            instance_counts[
                "test"
            ][class_id]
        )

        total = train + val + test

        print(
            f"{class_name:<20}"
            f"{train:>10}"
            f"{val:>10}"
            f"{test:>10}"
            f"{total:>10}"
        )


if __name__ == "__main__":
    count_dataset()