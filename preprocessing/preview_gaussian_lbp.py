from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


# Replace this with an actual ORIGINAL training image.
IMAGE_PATH = Path(
    r"C:\Users\ASUS\Downloads\l_light_01_missing_hole_01_1_600.jpg"
)

# Optional zoom around a known defect: (x1, y1, x2, y2).
# Coordinates must refer to the original image.
# Leave as None for the first run.
ROI = None

OUTPUT_DIR = Path(__file__).resolve().parent / "preview_outputs"

GAUSSIAN_SETTINGS = [
    ("No Gaussian filtering", None, None),
    ("Gaussian: 3x3, sigma=0.5", 3, 0.5),
    ("Gaussian: 5x5, sigma=1.0", 5, 1.0),
    ("Gaussian: 7x7, sigma=1.5", 7, 1.5),
]

LBP_SETTINGS = [
    (8, 1),
    (16, 2),
    (24, 3),
]


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image. Check IMAGE_PATH:\n{IMAGE_PATH}"
        )

    # Keep the original resolution and uint8 pixel values.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(f"Image dimensions: {width} x {height}")

    if ROI is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = ROI
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("ROI must be inside the original image.")

    # Four Gaussian settings, each with three LBP settings.
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))

    for row, (name, kernel, sigma) in enumerate(GAUSSIAN_SETTINGS):
        if kernel is None:
            filtered = gray.copy()
        else:
            filtered = cv2.GaussianBlur(
                gray,
                (kernel, kernel),
                sigmaX=sigma,
                sigmaY=sigma,
            )

        axes[row, 0].imshow(
            filtered[y1:y2, x1:x2],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axes[row, 0].set_title(name, fontsize=10)

        for column, (points, radius) in enumerate(LBP_SETTINGS, start=1):
            # Process the full image BEFORE cropping the preview.
            lbp = local_binary_pattern(
                filtered,
                P=points,
                R=radius,
                method="uniform",
            )

            axes[row, column].imshow(
                lbp[y1:y2, x1:x2],
                cmap="gray",
                vmin=0,
                vmax=points + 1,
                interpolation="nearest",
            )
            axes[row, column].set_title(
                f"Uniform LBP: P={points}, R={radius}",
                fontsize=10,
            )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        f"Gaussian + LBP preview: {IMAGE_PATH.name}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if ROI is None else "roi"
    output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_{suffix}_preview.png"

    fig.savefig(output_path, dpi=180)
    print(f"Preview saved to: {output_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()