import enum
import time
from dataclasses import dataclass
from threading import Thread
from queue import Queue

from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo


class TelemetryType(enum.Enum):
    T_ENTRY = 0
    T_LUMA = 1


@dataclass
class EntryTelemetry:
    timestamp: str
    direction: str

    def as_dict(self):
        return {"timestamp": self.timestamp, "direction": self.direction}


@dataclass
class LumaTelemetry:
    timemstamp: str
    mean_luma: float
    geom_mean_luma: float
    mean_lightness: float
    geom_mean_lightness: float
    median_lightness: float
    mean_filtered_lightness: float
    geom_mean_filtered_lightness: float
    median_filtered_lightness: float

    def as_dict(self):
        return {
            "timestamp": self.timemstamp,
            "mean luma": self.mean_luma,
            "geometric mean luma": self.geom_mean_luma,
            "mean lightness": self.mean_lightness,
            "geometric mean lightness": self.geom_mean_lightness,
            "median lightness": self.median_lightness,
            "mean filtered lightness": self.mean_filtered_lightness,
            "geometric mean filtered lightness": self.geom_mean_filtered_lightness,
            "median filtered lightness": self.median_filtered_lightness
        }


@dataclass
class TelemetryWrapper:
    type: TelemetryType
    telemetry: EntryTelemetry | LumaTelemetry


class TelemetrySender:
    def __init__(self, host: str, entry_at: str, luma_at: str, telemetry_queue: Queue):
        self.entry_client = TBDeviceMqttClient(
            host=host, username=entry_at)
        self.entry_client.connect()

        self.luma_client = TBDeviceMqttClient(
            host=host, username=luma_at)
        self.luma_client.connect()

        self.telemetry_queue: Queue = telemetry_queue

        self.thread = Thread(target=self.send, daemon=False)
        self.thread.start()

    def __del__(self):
        self.entry_client.disconnect()
        self.luma_client.disconnect()

    def send(self):
        while True:
            if not self.telemetry_queue.empty():
                wrapped = self.telemetry_queue.get()
                response = None
                if wrapped.type == TelemetryType.T_ENTRY:
                    response = self.entry_client.send_telemetry(wrapped.telemetry.as_dict())
                elif wrapped.type == TelemetryType.T_LUMA:
                    response = self.luma_client.send_telemetry(wrapped.telemetry.as_dict())
                if response == None:
                    raise Exception("Bad telemetry type")
                if response.get() != TBPublishInfo.TB_ERR_SUCCESS:
                    Exception("Telemetry was not sent")
            time.sleep(0.1)
            