import os
import configparser
from enum import Enum


class TColor(Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_config(config: str) -> configparser.ConfigParser:
    assert config is not None, "No configuration provided... See README.md"

    if not os.path.exists(config):
        raise Exception(f'Config file {config} does not exist')

    conf: configparser.ConfigParser = configparser.ConfigParser()

    if not conf.read(config):
        raise Exception(f"Config file {config} cant be read")

    return conf
