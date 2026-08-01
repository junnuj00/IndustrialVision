import cv2
import time

from config import Config

from core.camera import Camera
from core.detector import Detector
from core.preprocessor import Preprocessor

from utils.fps import FPSCounter
from utils.csv_writer import CSVWriter
from utils.image_saver import ImageSaver


def main():

    # Camera 생성
    camera = Camera(Config.CAMERA_INDEX)

    if not camera.is_opened():
        print("Camera Open Failed")
        return


    # YOLO Detector 생성
    detector = Detector(
        Config.MODEL_PATH,
        Config.CONF_THRESHOLD,
        Config.IOU_THRESHOLD
    )


    # Preprocessor 생성
    preprocessor = Preprocessor()


    # 현재 전처리 모드
    preprocess_mode = Config.DEFAULT_PREPROCESS_MODE


    # FPS Counter 생성
    fps_counter = FPSCounter()


    # CSV Writer 생성
    csv_writer = CSVWriter()


    # Image Saver 생성
    image_saver = ImageSaver(
        Config.IMAGE_SAVE_DIR
    )


    # 마지막 이미지 저장 시간
    last_save_time = 0


    while True:

        # 프레임 읽기
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break



        # =========================
        # 키 입력 전처리 선택
        # =========================

        if preprocess_mode == "clahe":

            processed_frame = preprocessor.clahe(frame)


        elif preprocess_mode == "gaussian":

            processed_frame = preprocessor.gaussian_blur(frame)


        elif preprocess_mode == "histogram":

            processed_frame = preprocessor.histogram_equalization(frame)


        else:

            processed_frame = frame



        # YOLO 추론
        results = detector.detect(processed_frame)



        # FPS 계산
        fps = fps_counter.update()



        # Detection 여부
        has_detection = False



        # CSV 저장
        for result in results:

            for box in result.boxes:

                has_detection = True

                class_id = int(box.cls[0])

                object_name = result.names[class_id]

                confidence = float(box.conf[0])


                csv_writer.write(
                    object_name,
                    confidence,
                    fps
                )



        # Bounding Box 이미지 생성
        annotated = detector.draw(results)



        # 이미지 자동 저장

        current_time = time.time()


        if has_detection and (
            current_time - last_save_time
            >= Config.SAVE_INTERVAL
        ):

            image_saver.save(annotated)

            last_save_time = current_time



        # FPS 표시

        cv2.putText(
            annotated,
            f"FPS : {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            Config.TEXT_COLOR,
            Config.FONT_THICKNESS
        )


        # 전처리 모드 표시

        cv2.putText(
            annotated,
            f"Mode : {preprocess_mode}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            Config.TEXT_COLOR,
            Config.FONT_THICKNESS
        )



        # 화면 출력

        cv2.imshow(
            Config.WINDOW_NAME,
            annotated
        )



        # =========================
        # Keyboard Control
        # =========================

        key = cv2.waitKey(1) & 0xFF



        if key == ord("1"):

            preprocess_mode = "original"

            print("Preprocess : Original")



        elif key == ord("2"):

            preprocess_mode = "gaussian"

            print("Preprocess : Gaussian Blur")



        elif key == ord("3"):

            preprocess_mode = "clahe"

            print("Preprocess : CLAHE")



        elif key == ord("4"):

            preprocess_mode = "histogram"

            print("Preprocess : Histogram Equalization")



        elif key == ord("q"):

            break




    # 종료 처리

    camera.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()