import cv2

from config import Config

from core.camera import Camera
from core.detector import Detector
from utils.fps import FPSCounter


def main():

    camera = Camera(Config.CAMERA_INDEX)

    if not camera.is_opened():
        print("Camera Open Failed")
        return

    detector = Detector(
        Config.MODEL_PATH,
        Config.CONF_THRESHOLD,
        Config.IOU_THRESHOLD
    )

    fps_counter = FPSCounter()

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to read frame.")
            break

        # YOLO 추론
        results = detector.detect(frame)

        # Bounding Box가 그려진 이미지 생성
        annotated = detector.draw(results)

        # FPS 계산
        fps = fps_counter.update()

        # FPS 출력
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
        cv2.imshow(Config.WINDOW_NAME, annotated)

        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()