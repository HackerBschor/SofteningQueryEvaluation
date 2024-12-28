from typing import List, Iterator

from operators import Operator


class Dummy(Operator):
    def __init__(self, name: str, columns: List[str], data: List[dict | tuple], num_tuples = 10):
        super().__init__(name, columns, num_tuples)
        self.data: Iterator = map(self._convert, data)


    def __str__(self) -> str:
        return f'Dummy({self.name})'

    def __next__(self) -> dict:
        return next(self.data)

    def next(self) -> List[dict]:
        idx = 0
        data = []

        try:
            while idx < self.num_tuples:
                data.append(next(self.data))
                idx += 1
        except StopIteration:
            pass

        return None if len(data) == 0 else data

    def _convert(self, r: dict | tuple):
        if isinstance(r, dict):
            return r
        else:
            return {self.columns[i]: value for i, value in enumerate(r)}