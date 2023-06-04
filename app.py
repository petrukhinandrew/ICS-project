import time
from queue import Queue
import json

import cv2
import imutils
from imutils.video import FPS, VideoStream

from luma_device import LumaDevice
from entry_tracker import EntryTracker
from telemetry import TelemetrySender

LUMA_DELAY = 120


class App:
    def __init__(self) -> None:
        self.telemetry_queue = Queue()
        self.__setup_config()
        self.__setup_video()
        self.__setup_tracker()
        self.__setup_luma_device()
        self.__setup_telemetry_sender()

    def __setup_config(self):
        with open("config.json", "r") as cfg:
            self.config = json.load(cfg)

    def __setup_video(self):
        # self.vs = VideoStream(config.url).start()
        self.vs = cv2.VideoCapture(self.config['cam_addr'])
        while self.__get_frame() is None:
            continue
        self.frame_height, self.frame_width = self.__get_frame().shape[:2]
        self.fps = FPS().start()
        self.total_frames = 0

    def __setup_tracker(self):
        self.entry_tracker = EntryTracker(
            self.frame_width, self.frame_height, self.config['net_confidence'], self.telemetry_queue)

    def __setup_luma_device(self):
        self.luma_frame_queue = Queue()
        self.luma_device = LumaDevice(
            self.luma_frame_queue, self.telemetry_queue)
        self.luma_last_frame = time.time()

    def __setup_telemetry_sender(self):
        self.telemetry_sender = TelemetrySender(
            self.config['host'], self.config['entry_at'], self.config['luma_at'], self.telemetry_queue)

    def __get_frame(self):
        frame = self.vs.read()[1]
        return None if frame is None else imutils.resize(frame, width=500)

    def run(self):
        while 1:
            frame = self.__get_frame()

            if time.time() - self.luma_last_frame > LUMA_DELAY:
                self.luma_frame_queue.put(frame.copy())
                self.luma_last_frame = time.time()

            coords = self.entry_tracker.process(
                frame, self.total_frames % self.config['skip_frames'] == 0)

            for (x, y) in coords:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.line(frame, (0, self.frame_height // 2),
                     (self.frame_width, self.frame_height // 2), (0, 0, 0), 3)
            cv2.imshow("lolkek", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            self.total_frames += 1
            self.fps.update()

    def __del__(self):
        self.fps.stop()
        print(self.fps.fps(), self.fps.elapsed())
        cv2.destroyAllWindows()
        # self.vs.stop()


if __name__ == "__main__":
    app = App()
    app.run()
