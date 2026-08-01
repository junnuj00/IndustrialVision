import cv2


class Preprocessor:

    def __init__(self):
        pass


    def gaussian_blur(self, frame):

        return cv2.GaussianBlur(
            frame,
            (5, 5),
            0
        )


    def clahe(self, frame):

        # BGR → LAB 변환
        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)


        # CLAHE 생성
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )


        # 밝기 채널 개선
        cl = clahe.apply(l)


        # 다시 합치기
        merged = cv2.merge(
            (cl, a, b)
        )


        # LAB → BGR
        result = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR
        )


        return result



    def histogram_equalization(self, frame):

        # BGR → YCrCb
        ycrcb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2YCrCb
        )


        y, cr, cb = cv2.split(ycrcb)


        # 밝기 채널 equalization
        y = cv2.equalizeHist(y)


        merged = cv2.merge(
            (y, cr, cb)
        )


        result = cv2.cvtColor(
            merged,
            cv2.COLOR_YCrCb2BGR
        )


        return result