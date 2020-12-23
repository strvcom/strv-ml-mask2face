import os
import json
from dataclasses import dataclass


@dataclass
class Configuration:
    def __init__(self, config_file_path: str = "configuration.json"):
        self.config_file_path = config_file_path
        self.config_json = None
        if os.path.exists(config_file_path):
            with open(self.config_file_path, 'r') as json_file:
                self.config_json = json.load(json_file)
        else:
            print(f'ERROR: Configuration JSON {config_file_path} does not exist.')

    def get(self, key: str):
        if key in self.config_json:
            return self.config_json[key]
        else:
            print(f'ERROR: Key \'{key}\' is not in configuration JSON.')
            return None
