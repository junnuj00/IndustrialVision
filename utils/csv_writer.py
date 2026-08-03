import csv
import os



class CSVWriter:


    def __init__(self):

        self.file_path = (
            "outputs/detection_results.csv"
        )


        # =========================
        # Output Folder
        # =========================

        os.makedirs(
            "outputs",
            exist_ok=True
        )



        # =========================
        # CSV Initialize
        # =========================

        with open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:


            writer = csv.writer(file)


            writer.writerow(
                [
                    "Frame",
                    "Object",
                    "Confidence",
                    "FPS"
                ]
            )



    # =========================
    # Detection Save
    # =========================

    def write(
        self,
        frame,
        object_name,
        confidence,
        fps
    ):


        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:


            writer = csv.writer(file)


            writer.writerow(
                [
                    frame,

                    object_name,

                    round(
                        confidence,
                        3
                    ),

                    round(
                        fps,
                        2
                    )
                ]
            )