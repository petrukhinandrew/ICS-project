from queue import Queue
from threading import Thread

from luma_calculator import LumaCalculator
from telemetry import TelemetryType, TelemetryWrapper


class LumaDevice:
    def __init__(self, frame_queue, telemetry_queue) -> None:
        self.frame_queue: Queue = frame_queue
        self.telemetry_queue: Queue = telemetry_queue
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                telemetry = LumaCalculator.calculate(frame)
                self.telemetry_queue.put(TelemetryWrapper(
                    TelemetryType.T_LUMA, telemetry))
