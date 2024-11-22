from enum import Enum
from typing import Dict


class Operator:
    def __init__(self) -> None:
        pass

    def next(self) -> (dict, bool):
        """
        :returns:  dict (the record)
        :returns: bool (is data from real)
        """
        pass


class DataType(Enum):
    INT = 1
    TEXT = 2
    DATE = 3


class Col:
    def __init__(self, name: str, data_type: DataType, origin: str, value) -> None:
        self.name = name
        self.data_type: DataType = data_type
        self.origin: str = origin
        self.value = value

    def __str__(self):
        return f"{self.name}: {self.value} ({self.origin})"


class Row:
    def __init__(self, idx: int, cols: [None, Dict[str, Col]] = None) -> None:
        self.idx: int = idx
        self.cols: Dict[str, Col] = {} if cols is None else cols

    def add_col(self, col: Col) -> None:
        self.cols[col.name] = col

    def __str__(self):
        return f"{{ {", ".join(map(str, self.cols.values()))} }}"

    def __getitem__(self, key):
        return self.cols[key].value

    def __setitem__(self, key, value):
        # self.cols[key].value = value
        raise Exception("Setting a column is not allowed")

    def __delitem__(self, key):
        raise Exception("Deleting a column is not allowed")
