from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo
import enum
from dataclasses import dataclass


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
    def __init__(self, host, device_token):
        self.client = TBDeviceMqttClient(
            host=host, username=device_token)
        self.client.connect()

    def __del__(self):
        self.client.disconnect()

    def send(self, telemetry: EntryTelemetry | LumaTelemetry):
        response = self.client.send_telemetry(telemetry.as_dict())
        if response.get() != TBPublishInfo.TB_ERR_SUCCESS:
            raise Exception('Telemetry was not sent')
