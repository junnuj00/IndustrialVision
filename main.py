import cv2

from config import Config

from core.camera import Camera
from core.detector import Detector

from utils.fps import FPSCounter
from utils.csv_writer import CSVWriter


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


    # FPS Counter 생성
    fps_counter = FPSCounter()


    # CSV Writer 생성
    csv_writer = CSVWriter()


    while True:

        # 프레임 읽기
        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break


        # YOLO 추론
        results = detector.detect(frame)


        # FPS 계산
        fps = fps_counter.update()


        # Detection 결과 CSV 저장
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                object_name = result.names[class_id]

                confidence = float(box.conf[0])


                csv_writer.write(
                    object_name,
                    confidence,
                    fps
                )


        # Bounding Box 표시
        annotated = detector.draw(results)


        # FPS 화면 표시
        cv2.putText(
            annotated,
            f"FPS : {fps:.2f}",
            (20, 40),
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


        # q 입력 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # 종료 처리
    camera.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()