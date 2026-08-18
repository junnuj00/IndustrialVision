from pathlib import Path

from ultralytics import YOLO

from config import Config


# ============================================
# Evaluation Configuration
# ============================================

# "baseline", "lowlight", "clahe", or "gamma"
EVALUATION_MODE = "gamma"

IMAGE_SIZE = 640

BATCH_SIZE = 1

DEVICE = "cpu"


# ============================================
# Project Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)


# ============================================
# Dataset Configuration
# ============================================

if EVALUATION_MODE == "baseline":

    DATASET = "coco128.yaml"

    RESULT_NAME = "baseline"


elif EVALUATION_MODE == "lowlight":

    DATASET = str(
        PROJECT_ROOT
        / "datasets"
        / "coco128_lowlight.yaml"
    )

    RESULT_NAME = "lowlight"


elif EVALUATION_MODE == "clahe":

    DATASET = str(
        PROJECT_ROOT
        / "datasets"
        / "coco128_lowlight_clahe.yaml"
    )

    RESULT_NAME = "lowlight_clahe"


elif EVALUATION_MODE == "gamma":

    DATASET = str(
        PROJECT_ROOT
        / "datasets"
        / "coco128_lowlight_gamma.yaml"
    )

    RESULT_NAME = "lowlight_gamma"


else:

    raise ValueError(
        f"Unknown evaluation mode: "
        f"{EVALUATION_MODE}"
    )


# ============================================
# Evaluation
# ============================================

def evaluate():

    print("=" * 60)
    print("YOLO11 Object Detection Evaluation")
    print("=" * 60)

    print(
        f"Mode    : "
        f"{EVALUATION_MODE}"
    )

    print(
        f"Model   : "
        f"{Config.MODEL_PATH}"
    )

    print(
        f"Dataset : "
        f"{DATASET}"
    )

    print(
        f"Image   : "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Device  : "
        f"{DEVICE}"
    )

    print("=" * 60)


    model = YOLO(
        Config.MODEL_PATH
    )


    metrics = model.val(

        data=DATASET,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        project=str(OUTPUT_DIR),

        name=RESULT_NAME,

        plots=True,

        verbose=True

    )


    precision = metrics.box.mp

    recall = metrics.box.mr

    map50 = metrics.box.map50

    map50_95 = metrics.box.map


    print()

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(
        f"Mode          : "
        f"{EVALUATION_MODE}"
    )

    print(
        f"Precision     : "
        f"{precision:.4f}"
    )

    print(
        f"Recall        : "
        f"{recall:.4f}"
    )

    print(
        f"mAP50         : "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95      : "
        f"{map50_95:.4f}"
    )

    print("=" * 60)

    print(
        f"Results saved to: "
        f"{OUTPUT_DIR / RESULT_NAME}"
    )


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    evaluate()