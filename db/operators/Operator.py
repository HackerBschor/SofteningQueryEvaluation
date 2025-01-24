from typing import Iterator

from ..structure import SQLTable


class Operator(Iterator):
    def __init__(self, table: SQLTable, num_tuples: int) -> None:
        self.table: SQLTable = table
        self.num_tuples: int = num_tuples
        self.open()

    def __next__(self) -> dict:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def open(self) -> None:
        raise NotImplementedError()

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def get_description(self) -> str:
        return str(self)

    def get_structure(self) -> tuple[str, list] | str:
        return f"{id(self)}:{self.get_description()}"
