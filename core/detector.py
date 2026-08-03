from ultralytics import YOLO


class Detector:


    def __init__(
        self,
        model_path,
        conf=0.5,
        iou=0.7
    ):

        self.model = YOLO(
            model_path
        )

        self.conf = conf

        self.iou = iou



    def detect(self, frame):

        results = self.model(

            frame,

            conf=self.conf,

            iou=self.iou,

            imgsz=640,

            verbose=False

        )

        return results



    def draw(self, results):

        return results[0].plot()



    def get_boxes(self, results):

        return results[0].boxes