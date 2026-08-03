import cv2


class Preprocessor:


    def __init__(self):

        # CLAHE 객체 생성
        # 매 프레임마다 생성하지 않고 재사용
        self.clahe_processor = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )



    # =========================
    # Gaussian Blur
    # =========================

    def gaussian_blur(self, frame):

        return cv2.GaussianBlur(
            frame,
            (5, 5),
            0
        )



    # =========================
    # CLAHE
    # =========================

    def clahe(self, frame):

        # --------------------------------
        # 속도 개선을 위한 resize
        # --------------------------------

        small = cv2.resize(
            frame,
            (960, 540)
        )


        # BGR → LAB 변환

        lab = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2LAB
        )


        # L(밝기), A, B 채널 분리

        l, a, b = cv2.split(
            lab
        )


        # 밝기 채널에만 CLAHE 적용

        l = self.clahe_processor.apply(
            l
        )


        # 다시 합치기

        merged = cv2.merge(
            (
                l,
                a,
                b
            )
        )


        # LAB → BGR

        result = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR
        )


        # --------------------------------
        # 원본 영상 크기로 복원
        # --------------------------------

        result = cv2.resize(
            result,
            (
                frame.shape[1],
                frame.shape[0]
            )
        )


        return result



    # =========================
    # Histogram Equalization
    # =========================

    def histogram_equalization(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        equalized = cv2.equalizeHist(
            gray
        )


        result = cv2.cvtColor(
            equalized,
            cv2.COLOR_GRAY2BGR
        )


        return result