class Config:
    MODEL_PATH = "models/yolo11n.pt"

    CAMERA_INDEX = 0

    WINDOW_NAME = "Industrial Vision"

    FONT_SCALE = 1
    FONT_THICKNESS = 2
    TEXT_COLOR = (0, 255, 0)

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.7

    IMAGE_SAVE_DIR = "outputs/images"

    SAVE_INTERVAL = 1.0   # 초 단위