import cv2


class Camera:

    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()