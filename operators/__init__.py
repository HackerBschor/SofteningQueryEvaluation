from typing import Iterator, List, Any


class ColumnNotFoundException(Exception):
    def __init__(self, column: str, columns: List[str]) -> None:
        self.column: str = column
        self.columns: List[str] = columns

    def __str__(self) -> str:
        return f"Column '{self.column}' not found in parent columns {self.columns}"


class Operator(Iterator):
    def __init__(self, name: str, columns: List[str]) -> None:
        self.name: str = name
        self.columns: List[str] = columns

    def __next__(self) -> dict:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def get_structure(self) -> tuple[str, List] | str:
        return f"{id(self)}:{self}"


class Column:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def __str__(self) -> str:
        return str(self.name)

    def get(self, t: dict) -> Any:
        return t[self.name]

class Constant:
    def __init__(self, value: Any) -> None:
        self.value: Any = value

    def __str__(self) -> str:
        return f"'{self.value}'" if isinstance(self.value, str) else str(self.value)

    def get(self, _: Any) -> Any:
        return self.value