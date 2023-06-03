from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo
from enum import Enum
from dataclasses import dataclass
import datetime


class TelemetryType(Enum):
    T_ENTRY = 0
    T_LUMA = 1


@dataclass
class EntryTelemetry:
    timestamp: datetime.datetime
    direction: str

    def as_dict(self):
        pass


@dataclass
class LumaTelemetry:
    mean_luma: float
    geom_mean_luma: float
    mean_lightness: float
    geom_mean_lightness: float
    median_lightness: float
    mean_filtered_lightness: float
    geom_mean_filtered_lightness: float
    median_filtered_lightness: float

    def as_dict(self):
        pass


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

    def send(self, telemetry):
        sent = self.client.send_telemetry(telemetry)
        if sent != TBPublishInfo.TB_ERR_SUCCESS:
            raise Exception('Telemetry was not sent')
