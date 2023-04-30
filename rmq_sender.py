import datetime
import json
from datetime import time
import yaml

import asyncio
import rabbitmq


class RMQSender:
    def __init__(self, source_name, class_names=None):
        self.class_names = class_names
        self.source_name = source_name
        self.time_ticks = 0

    def _data_preprocessing(self, data):

        yaml_dict = {"detections": data[:, 4:].tolist(),
                     "device": self.source_name,
                     "classNames": self.class_names}
        yaml_string = yaml.dump(yaml_dict)
        print(f"The YAML string is: {yaml_string}")
        return yaml_string

    def send(self, data):
        json_string = self._data_preprocessing(data)
        rabbitmq.send_message(json_string)