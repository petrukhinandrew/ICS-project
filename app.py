from imutils.video import VideoStream
from imutils.video import FPS
import config
import imutils
from luma_device import LumaDevice
from entry_tracker import EntryTracker
from queue import Queue
from threading import Thread
# import cv2
import time
from telemetry import TelemetryType, TelemetrySender

LUMA_DELAY = 60


class App:
    def __init__(self) -> None:
        self.telemetry_queue = Queue()
        self.__setup_video()
        self.__setup_luma_device()
        self.__setup_tracker()
        self.__setup_telemetry_sender()

    def __setup_telemetry_sender(self):
        self.entry_sender = TelemetrySender(config.HOST, config.CAM_DT)
        self.luma_sender = TelemetrySender(config.HOST, config.LUMA_DT)

    def __setup_luma_device(self):
        self.luma_frame_queue = Queue()
        self.luma_device = LumaDevice(
            self.luma_frame_queue, self.telemetry_queue)
        self.luma_last_frame = time.time()
        self.luma_thread = Thread(target=self.luma_device.run, daemon=True)
        self.luma_thread.start()

    def __setup_tracker(self):
        self.entry_tracker = EntryTracker(
            self.frame_width, self.frame_height, config.Confidence, self.telemetry_queue)

    def __setup_video(self):
        self.vs = VideoStream(config.url).start()
        while self._get_frame() is None:
            continue
        self.frame_width, self.frame_height = self._get_frame().shape[:2]
        self.fps = FPS()
        self.total_frames = 0

    def _get_frame(self):
        frame = self.vs.read()
        return None if frame is None else imutils.resize(frame, width=500)

    def _send_telemetry(self):
        while not self.telemetry_queue.empty():
            wrapper = self.telemetry_queue.get()
            telemetry_type, telemetry = wrapper.type, wrapper.telemetry
            print(telemetry)
            if telemetry_type == TelemetryType.T_ENTRY:
                self.entry_sender.send(telemetry)
            elif telemetry_type == TelemetryType.T_LUMA:
                self.luma_sender.send(telemetry)
            else:
                raise Exception("Bad telemetry type: " + str(telemetry_type))
            print(telemetry)

    def run(self):
        while 1:
            frame = self._get_frame()

            # cv2.imshow("lolkek", frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

            if time.time() - self.luma_last_frame > LUMA_DELAY:
                self.luma_frame_queue.put(frame.copy())
                self.luma_last_frame = time.time()

            self.entry_tracker.process(
                frame, self.total_frames % config.SkipFrames == 0)

            self._send_telemetry()

        # cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()
