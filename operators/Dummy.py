from typing import List, Iterator

from operators import Operator


class Dummy(Operator):
    def __init__(self, name: str, columns: List[str], data: List[dict | tuple]):
        self.data: Iterator = iter(data)
        super().__init__(name, columns)

    def __str__(self):
        return f'Dummy({self.name})'

    def __next__(self) -> dict:
        r = next(self.data)

        if isinstance(r, dict):
            return r
        else:
            return {self.columns[i]: value for i, value in enumerate(r)}