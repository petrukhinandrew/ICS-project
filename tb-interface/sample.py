from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo
import json
from typing import Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ServerInfo:
    host: str
    port: str
    protocol: str
    threashold: int
    type: str 


def get_server_info_from_config(config_path: str, server_tag: str = "demoserver") -> Dict[str,str|int]:
    # NEED SAFETY
    config_file = open("config.json", 'r')
    parsed_config = json.load(config_file)
    config_file.close()

    raw_info = parsed_config[server_tag]

    return raw_info

server = get_server_info_from_config("config.json")

temperature = 12

telemetry = {"temperature": temperature}
#where to store AC?
client = TBDeviceMqttClient(host=server["host"], username="8JpXCiP482pz9y7tBF9t")

# Connect to ThingsBoard
client.connect()
# Sending telemetry without checking the delivery status
client.send_telemetry(telemetry) 
# Sending telemetry and checking the delivery status (QoS = 1 by default)
result = client.send_telemetry(telemetry)
# get is a blocking call that awaits delivery status  
success = result.get() == TBPublishInfo.TB_ERR_SUCCESS
print(success)
# Disconnect from ThingsBoard
client.disconnect()
