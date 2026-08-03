import cv2
import os

from config import Config

from core.camera import Camera
from core.detector import Detector
from core.preprocessor import Preprocessor

from utils.fps import FPSCounter
from utils.csv_writer import CSVWriter
from utils.metrics import Metrics



# =========================
# Preprocessing Mode Select
# =========================

def select_preprocess_mode():

    print()
    print("=" * 45)
    print(" Industrial Vision Inspection ")
    print("=" * 45)

    print()

    print(
        f"Input Mode : {Config.INPUT_MODE}"
    )

    if Config.INPUT_MODE == "video":

        print(
            f"Video : {os.path.basename(Config.VIDEO_PATH)}"
        )


    print()

    print(
        "Select Preprocessing Mode"
    )

    print()

    print(
        "[1] Original"
    )

    print(
        "[2] Gaussian Blur"
    )

    print(
        "[3] CLAHE"
    )

    print(
        "[4] Histogram Equalization"
    )


    while True:

        choice = input(
            "\n>> "
        )


        if choice == "1":

            return "original"


        elif choice == "2":

            return "gaussian"


        elif choice == "3":

            return "clahe"


        elif choice == "4":

            return "histogram"


        else:

            print(
                "Invalid Input. Select 1~4"
            )



def main():


    # =========================
    # Select Mode
    # =========================

    preprocess_mode = (
        select_preprocess_mode()
    )


    print()

    print(
        f"Selected : {preprocess_mode}"
    )


    print()



    # =========================
    # Camera
    # =========================

    camera = Camera()


    if not camera.is_opened():

        print(
            "Camera / Video Open Failed"
        )

        return



    # =========================
    # Detector
    # =========================

    detector = Detector(
        Config.MODEL_PATH,
        Config.CONF_THRESHOLD,
        Config.IOU_THRESHOLD
    )


    print(
        f"Model : {Config.MODEL_PATH}"
    )



    # =========================
    # Preprocessor
    # =========================

    preprocessor = Preprocessor()



    # =========================
    # Utils
    # =========================

    fps_counter = FPSCounter()

    csv_writer = CSVWriter()

    metrics = Metrics()



    # Frame Count

    frame_index = 0



    # =========================
    # Window
    # =========================

    cv2.namedWindow(
        Config.WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )


    cv2.resizeWindow(
        Config.WINDOW_NAME,
        Config.WINDOW_WIDTH,
        Config.WINDOW_HEIGHT
    )



    # =========================
    # Main Loop
    # =========================

    while True:


        ret, frame = camera.read()


        if not ret:

            break



        frame_index += 1



        # =========================
        # Preprocessing
        # =========================

        if preprocess_mode == "gaussian":

            processed_frame = (
                preprocessor.gaussian_blur(frame)
            )


        elif preprocess_mode == "clahe":

            processed_frame = (
                preprocessor.clahe(frame)
            )


        elif preprocess_mode == "histogram":

            processed_frame = (
                preprocessor.histogram_equalization(frame)
            )


        else:

            processed_frame = frame



        # =========================
        # Detection
        # =========================

        results = detector.detect(
            processed_frame
        )



        metrics.update(
            results
        )



        # =========================
        # FPS
        # =========================

        fps = (
            fps_counter.update()
        )



        # =========================
        # Detection CSV
        # =========================

        for result in results:


            for box in result.boxes:


                class_id = int(
                    box.cls[0]
                )


                object_name = (
                    result.names[class_id]
                )


                confidence = float(
                    box.conf[0]
                )


                csv_writer.write(
                    frame_index,
                    object_name,
                    confidence,
                    fps
                )



        # =========================
        # Draw
        # =========================

        annotated = detector.draw(
            results
        )



        # =========================
        # Display
        # =========================

        cv2.putText(
            annotated,
            f"FPS : {fps:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            Config.TEXT_COLOR,
            Config.FONT_THICKNESS
        )


        cv2.putText(
            annotated,
            f"Mode : {preprocess_mode}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            Config.TEXT_COLOR,
            Config.FONT_THICKNESS
        )


        cv2.putText(
            annotated,
            f"Frame : {frame_index}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            Config.TEXT_COLOR,
            Config.FONT_THICKNESS
        )



        cv2.imshow(
            Config.WINDOW_NAME,
            annotated
        )



        # =========================
        # Quit
        # =========================

        key = cv2.waitKey(1) & 0xff


        if key == ord("q"):

            break



    # =========================
    # Final Metrics
    # =========================

    avg_fps = (
        fps_counter.get_average()
    )


    result = metrics.get_result()



    print()

    print(
        "=" * 35
    )

    print(
        " Final Metrics "
    )

    print(
        "=" * 35
    )


    print(
        f"Mode : {preprocess_mode}"
    )


    print(
        f"Frames : {result['Frames']}"
    )


    print(
        f"Detection Count : {result['Detection Count']}"
    )


    print(
        f"Average Confidence : {result['Average Confidence']}"
    )


    print(
        f"Average FPS : {avg_fps:.2f}"
    )



    metrics.save_csv(
        preprocess_mode,
        avg_fps
    )



    camera.release()

    cv2.destroyAllWindows()



    print()

    print(
        "Performance Saved"
    )


    print(
        "outputs/metrics/performance.csv"
    )


    print(
        "Detection Saved"
    )


    print(
        "outputs/detection_results.csv"
    )




if __name__ == "__main__":

    main()