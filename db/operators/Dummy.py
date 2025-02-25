import copy

from typing import Iterator

from db.structure import SQLTable, SQLColumn
from db.operators.Operator import Operator


class Dummy(Operator):
    def __init__(self, name: str, columns: list[str] | list[SQLColumn], data: list[dict | tuple], num_tuples=10):
        self.data: list[dict] = list(map(lambda r: self._convert_function(r, columns), copy.deepcopy(data)))

        self.idx: int | None = None
        self.iter: Iterator[dict] | None = None

        if isinstance(columns[0], str):
         columns = [SQLColumn(col, str(type(self.data[0][col]))) for col in columns]

        table = SQLTable(None, name, columns)
        super().__init__(table, num_tuples)

    def __str__(self) -> str:
        return self.table.table_name

    def __next__(self) -> dict:
        try:
            return next(self.iter)
        except StopIteration:
            self.close()
            raise StopIteration

    def open(self) -> Operator:
        # Reset Iterator
        self.iter: Iterator[dict] | None = iter(self.data)
        self.idx: int | None = 0
        return self

    def next_vectorized(self) -> list[dict]:
        if self.idx is None:
            raise StopIteration

        idx: int = self.idx
        self.idx += 1

        # Return next "slice"
        return self.data[idx * self.num_tuples: min((idx + 1) * self.num_tuples, len(self.data))]

    def close(self) -> None:
        self.iter: Iterator[dict] | None = None
        self.idx: int | None = None

    @staticmethod
    def _convert_function(r, columns):
        return r if isinstance(r, dict) else {columns[i]: value for i, value in enumerate(r)}
