import cv2
from pathlib import Path


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to the lightness channel of a colour image.
    """

    # Convert BGR to LAB colour space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # Apply CLAHE only to the lightness channel
    enhanced_l = clahe.apply(l_channel)

    # Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


if __name__ == "__main__":

    # CHANGE this filename to one actual image in your dataset
    input_path = Path(
        "dataset/train/images/l_light_01_missing_hole_01_1_600.jpg"
    )

    output_path = Path(
        "results/clahe_sample.jpg"
    )

    # Create results folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {input_path}"
        )

    # Apply CLAHE
    enhanced_image = apply_clahe(
        image,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    )

    # Save result
    cv2.imwrite(
        str(output_path),
        enhanced_image
    )

    print(f"Original image: {input_path}")
    print(f"CLAHE image saved to: {output_path}")