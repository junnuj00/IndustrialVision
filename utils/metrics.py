import csv
import os


class Metrics:

    def __init__(self):

        self.total_frames = 0

        self.total_detections = 0

        self.total_confidence = 0.0



    def update(self, results):

        self.total_frames += 1


        for result in results:

            for box in result.boxes:

                self.total_detections += 1


                confidence = float(
                    box.conf[0]
                )


                self.total_confidence += confidence



    def get_result(self):

        if self.total_detections == 0:

            avg_confidence = 0

        else:

            avg_confidence = (
                self.total_confidence /
                self.total_detections
            )


        return {

            "Frames":
            self.total_frames,


            "Detection Count":
            self.total_detections,


            "Average Confidence":
            round(
                avg_confidence,
                3
            )

        }



    # =========================
    # Metrics 초기화
    # =========================

    def reset(self):

        self.total_frames = 0

        self.total_detections = 0

        self.total_confidence = 0.0



    # =========================
    # CSV 저장
    # =========================

    def save_csv(
        self,
        mode,
        avg_fps
    ):

        result = self.get_result()


        os.makedirs(
            "outputs/metrics",
            exist_ok=True
        )


        file_path = (
            "outputs/metrics/performance.csv"
        )


        file_exists = os.path.exists(
            file_path
        )


        with open(
            file_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:


            writer = csv.writer(f)


            if not file_exists:

                writer.writerow(
                    [
                        "Mode",
                        "Frames",
                        "Detection Count",
                        "Average Confidence",
                        "Average FPS"
                    ]
                )


            writer.writerow(
                [
                    mode,

                    result["Frames"],

                    result["Detection Count"],

                    result["Average Confidence"],

                    round(
                        avg_fps,
                        2
                    )
                ]
            )