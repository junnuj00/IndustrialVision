class Config:

    # =========================
    # Input
    # =========================

    # "camera" or "video"
    INPUT_MODE = "video"

    # Camera
    CAMERA_INDEX = 0

    # Video
    VIDEO_PATH = "inputs/videos/city.mp4"



    # =========================
    # YOLO Model
    # =========================

    MODEL_PATH = "models/yolo11n.pt"

    CONF_THRESHOLD = 0.25

    IOU_THRESHOLD = 0.45



    # =========================
    # OpenCV Window
    # =========================

    WINDOW_NAME = (
        "Industrial Vision Inspection"
    )

    WINDOW_WIDTH = 960

    WINDOW_HEIGHT = 540



    # =========================
    # Font
    # =========================

    FONT_SCALE = 0.8

    FONT_THICKNESS = 2

    TEXT_COLOR = (
        0,
        255,
        0
    )



    # =========================
    # Image Save
    # =========================

    SAVE_INTERVAL = 1.0