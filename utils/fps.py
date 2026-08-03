import time


class FPSCounter:

    def __init__(self):

        self.reset()


    # =========================
    # Reset
    # =========================

    def reset(self):

        self.prev_time = time.perf_counter()

        self.total_fps = 0.0

        self.count = 0


    # =========================
    # Current FPS
    # =========================

    def update(self):

        current_time = time.perf_counter()

        fps = 1 / (
            current_time - self.prev_time
        )

        self.prev_time = current_time

        self.total_fps += fps

        self.count += 1

        return fps


    # =========================
    # Average FPS
    # =========================

    def get_average(self):

        if self.count == 0:

            return 0

        return (
            self.total_fps /
            self.count
        )