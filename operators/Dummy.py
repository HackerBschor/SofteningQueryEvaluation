import copy

from typing import Iterator

from operators import Operator


class Dummy(Operator):
    def __init__(self, name: str, columns: list[str], data: list[dict | tuple], num_tuples = 10):
        convert_function = lambda r: r if isinstance(r, dict) else {columns[i]: value for i, value in enumerate(r)}
        self.data: list[dict] = list(map(convert_function, copy.deepcopy(data)))

        self.idx: int | None = None
        self.iter: Iterator[dict] | None = None
        super().__init__(name, columns, num_tuples)


    def __str__(self) -> str:
        return f'Dummy({self.name})'

    def __next__(self) -> dict:
        try:
            return next(self.iter)
        except StopIteration:
            self.close()
            raise StopIteration

    def open(self) -> None:
        self.iter: Iterator[dict] | None = iter(self.data) # Reset Iterator
        self.idx: int | None = 0

    def next_vectorized(self) -> list[dict]:
        if self.idx is None:
            raise StopIteration

        idx: int = self.idx
        self.idx += 1

        # Return next "slice"
        return self.data[idx * self.num_tuples : min((idx + 1) * self.num_tuples, len(self.data))]


    def close(self) -> None:
        self.iter: Iterator[dict] | None = None
        self.idx: int | None = None