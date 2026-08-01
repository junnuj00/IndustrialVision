import cv2
import os
from datetime import datetime


class ImageSaver:

    def __init__(self, save_dir="outputs/images"):

        self.save_dir = save_dir

        # 저장 폴더 생성
        os.makedirs(self.save_dir, exist_ok=True)


    def save(self, image):

        # 현재 시간을 파일 이름으로 사용
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"

        save_path = os.path.join(
            self.save_dir,
            filename
        )

        cv2.imwrite(
            save_path,
            image
        )

        print(f"[Saved] {save_path}")