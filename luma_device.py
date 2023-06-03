from luma_calculator import LumaCalculator
from queue import Queue

class LumaDevice:
    def __init__(self, frame_queue, telemetry_queue) -> None:
        self.frame_queue: Queue = frame_queue
        self.telemetry_queue: Queue = telemetry_queue
    def run(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                telemetry = LumaCalculator.calculate(frame)
                self.telemetry_queue.put(telemetry)