import time
from queue import Queue

import cv2
import imutils
from imutils.video import FPS, VideoStream

import config
from luma_device import LumaDevice
from entry_tracker import EntryTracker
from telemetry import TelemetryType, TelemetrySender

LUMA_DELAY = 120


class App:
    def __init__(self) -> None:
        self.telemetry_queue = Queue()
        self.__setup_video()
        self.__setup_luma_device()
        self.__setup_tracker()
        self.__setup_telemetry_sender()

    def __setup_telemetry_sender(self):
        self.entry_sender = TelemetrySender(config.HOST, config.ENTRY_AT)
        self.luma_sender = TelemetrySender(config.HOST, config.LUMA_AT)

    def __setup_luma_device(self):
        self.luma_frame_queue = Queue()
        self.luma_device = LumaDevice(
            self.luma_frame_queue, self.telemetry_queue)
        self.luma_last_frame = time.time()

    def __setup_tracker(self):
        self.entry_tracker = EntryTracker(
            self.frame_width, self.frame_height, config.Confidence, self.telemetry_queue)

    def __setup_video(self):
        # self.vs = VideoStream(config.url).start()
        self.vs = cv2.VideoCapture(config.url)
        while self.__get_frame() is None:
            continue
        self.frame_width, self.frame_height = self.__get_frame().shape[:2]
        self.fps = FPS().start()
        self.total_frames = 0

    def __get_frame(self):
        frame = self.vs.read()[1]
        return None if frame is None else imutils.resize(frame, width=500)

    def __send_telemetry(self):
        while not self.telemetry_queue.empty():
            wrapper = self.telemetry_queue.get()
            telemetry_type, telemetry = wrapper.type, wrapper.telemetry
            if telemetry_type == TelemetryType.T_ENTRY:
                self.entry_sender.send(telemetry)
            elif telemetry_type == TelemetryType.T_LUMA:
                self.luma_sender.send(telemetry)
            else:
                raise Exception("Bad telemetry type: " + str(telemetry_type))

    def run(self):
        while 1:
            frame = self.__get_frame()

            cv2.imshow("lolkek", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if time.time() - self.luma_last_frame > LUMA_DELAY:
                self.luma_frame_queue.put(frame.copy())
                self.luma_last_frame = time.time()

            self.entry_tracker.process(
                frame, self.total_frames % config.SkipFrames == 0)

            self.__send_telemetry()

            self.total_frames += 1
            self.fps.update()

        

    def __del__(self):
        self.fps.stop()
        print(self.fps.fps(), self.fps.elapsed())
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    app.run()
