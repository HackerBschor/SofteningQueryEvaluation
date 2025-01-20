import os
import configparser
import numpy as np

class TColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Measure:
    def __init__(self, is_distance: bool):
        self.is_distance: bool = is_distance
        self.similarity: float | None = None

    def __call__(self, u: np.array, v: np.array, threshold: float | None = None) -> float | bool:
        self.similarity: float = self._calc(u, v)
        if threshold is not None:
            return self.similarity <= threshold if self.is_distance else self.similarity >= threshold
        else:
            return self.similarity

    def _calc(self, u: np.array, v: np.array) -> float:
        raise NotImplementedError()

class CosineSimilarity(Measure):
    def __init__(self):
        super().__init__(False)

    def _calc(self, u: np.array, v: np.array) -> float:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class EuclidianDistance(Measure):
    def __init__(self):
        super().__init__(True)

    def _calc(self, u: np.array, v: np.array):
        return np.linalg.norm(u - v)


def get_config(config: str) -> configparser.ConfigParser:
    assert config is not None, "No configuration provided... See README.md"

    if not os.path.exists(config):
        raise Exception(f'Config file {config} does not exist')

    conf: configparser.ConfigParser = configparser.ConfigParser()

    if not conf.read(config):
        raise Exception(f"Config file {config} cant be read")

    return conf