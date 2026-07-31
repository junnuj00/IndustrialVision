import csv
import os
from datetime import datetime


class CSVWriter:

    def __init__(self, save_path="outputs/detection_results.csv"):

        self.save_path = save_path

        # outputs 폴더 없으면 생성
        os.makedirs("outputs", exist_ok=True)

        # CSV 파일이 없으면 헤더 생성
        if not os.path.exists(self.save_path):

            with open(
                self.save_path,
                "w",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "Time",
                        "Object",
                        "Confidence",
                        "FPS"
                    ]
                )


    def write(
        self,
        object_name,
        confidence,
        fps
    ):

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            self.save_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    current_time,
                    object_name,
                    confidence,
                    fps
                ]
            )