from pathlib import Path
import shutil

import cv2
import numpy as np


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "coco128"
)

SOURCE_IMAGES = (
    SOURCE_DATASET
    / "images"
    / "train2017"
)

SOURCE_LABELS = (
    SOURCE_DATASET
    / "labels"
    / "train2017"
)


OUTPUT_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "coco128_lowlight"
)

OUTPUT_IMAGES = (
    OUTPUT_DATASET
    / "images"
    / "train2017"
)

OUTPUT_LABELS = (
    OUTPUT_DATASET
    / "labels"
    / "train2017"
)


# ============================================
# Low-light Configuration
# ============================================

# 1.0 = original brightness
# Smaller value = darker image
BRIGHTNESS_FACTOR = 0.2


# ============================================
# Low-light Transformation
# ============================================

def make_low_light(image, factor=BRIGHTNESS_FACTOR):

    dark_image = image.astype(
        np.float32
    )

    dark_image *= factor

    dark_image = np.clip(
        dark_image,
        0,
        255
    )

    return dark_image.astype(
        np.uint8
    )


# ============================================
# Dataset Generation
# ============================================

def create_low_light_dataset():

    print("=" * 60)
    print("Creating COCO128 Low-light Dataset")
    print("=" * 60)

    print(
        f"Source : {SOURCE_IMAGES}"
    )

    print(
        f"Output : {OUTPUT_IMAGES}"
    )

    print(
        f"Brightness factor : "
        f"{BRIGHTNESS_FACTOR}"
    )

    print("=" * 60)


    # ----------------------------------------
    # Check source dataset
    # ----------------------------------------

    if not SOURCE_IMAGES.exists():

        raise FileNotFoundError(
            f"Source image directory not found: "
            f"{SOURCE_IMAGES}"
        )


    if not SOURCE_LABELS.exists():

        raise FileNotFoundError(
            f"Source label directory not found: "
            f"{SOURCE_LABELS}"
        )


    # ----------------------------------------
    # Create output directories
    # ----------------------------------------

    OUTPUT_IMAGES.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_LABELS.mkdir(
        parents=True,
        exist_ok=True
    )


    # ----------------------------------------
    # Find images
    # ----------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_paths = [

        path

        for path in SOURCE_IMAGES.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in image_extensions
        )
    ]


    if not image_paths:

        raise RuntimeError(
            "No images found in source dataset."
        )


    print(
        f"Images found : "
        f"{len(image_paths)}"
    )


    # ----------------------------------------
    # Process images
    # ----------------------------------------

    processed = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            print(
                f"Skipped unreadable image: "
                f"{image_path.name}"
            )

            continue


        low_light = make_low_light(
            image
        )


        output_path = (
            OUTPUT_IMAGES
            / image_path.name
        )


        success = cv2.imwrite(
            str(output_path),
            low_light
        )


        if not success:

            print(
                f"Failed to save: "
                f"{output_path}"
            )

            continue


        processed += 1


    # ----------------------------------------
    # Copy labels
    # ----------------------------------------

    label_paths = list(
        SOURCE_LABELS.glob("*.txt")
    )


    for label_path in label_paths:

        shutil.copy2(
            label_path,
            OUTPUT_LABELS
            / label_path.name
        )


    # ----------------------------------------
    # Result
    # ----------------------------------------

    print()
    print("=" * 60)
    print("Low-light Dataset Created")
    print("=" * 60)

    print(
        f"Processed images : "
        f"{processed}"
    )

    print(
        f"Copied labels    : "
        f"{len(label_paths)}"
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DATASET}"
    )

    print("=" * 60)


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    create_low_light_dataset()