from pathlib import Path

import cv2

from config import Config


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / Config.VIDEO_PATH
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vlm"
)

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "city_frame.jpg"
)


# ============================================
# Frame Extraction
# ============================================

def extract_frame():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not capture.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{VIDEO_PATH}"
        )


    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # 영상 중간 프레임 사용
    target_frame = (
        total_frames // 2
    )


    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame
    )


    success, frame = capture.read()


    capture.release()


    if not success:

        raise RuntimeError(
            "Failed to extract frame."
        )


    cv2.imwrite(
        str(OUTPUT_IMAGE),
        frame
    )


    print("=" * 60)
    print("Frame Extraction Complete")
    print("=" * 60)

    print(
        f"Total Frames : "
        f"{total_frames}"
    )

    print(
        f"Target Frame : "
        f"{target_frame}"
    )

    print(
        f"Saved Image  : "
        f"{OUTPUT_IMAGE}"
    )

    print("=" * 60)


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    extract_frame()