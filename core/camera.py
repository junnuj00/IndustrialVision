import cv2

from config import Config


class Camera:

    def __init__(self):

        # =========================
        # Camera Input
        # =========================

        if Config.INPUT_MODE == "camera":

            self.cap = cv2.VideoCapture(
                Config.CAMERA_INDEX
            )

            print(
                "[INFO] Input Mode : Camera"
            )


        # =========================
        # Video Input
        # =========================

        elif Config.INPUT_MODE == "video":

            self.cap = cv2.VideoCapture(
                Config.VIDEO_PATH
            )

            print(
                "[INFO] Input Mode : Video"
            )


        else:

            raise ValueError(
                "Invalid INPUT_MODE"
            )


    # =========================
    # Check Input
    # =========================

    def is_opened(self):

        return self.cap.isOpened()


    # =========================
    # Read Frame
    # =========================

    def read(self):

        return self.cap.read()


    # =========================
    # Release
    # =========================

    def release(self):

        if self.cap is not None:

            self.cap.release()