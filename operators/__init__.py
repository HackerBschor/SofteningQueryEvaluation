from typing import Iterator, List, Any

from utils import Measure, CosineSimilarity
from utils.Model import EmbeddingModel


class ColumnNotFoundException(Exception):
    def __init__(self, column: str, columns: List[str]) -> None:
        self.column: str = column
        self.columns: List[str] = columns

    def __str__(self) -> str:
        return f"Column '{self.column}' not found in parent columns {self.columns}"


class Operator(Iterator):
    def __init__(self, name: str, columns: List[str], num_tuples: int) -> None:
        self.name: str = name
        self.columns: List[str] = columns
        self.num_tuples: int = num_tuples
        self.open()

    def __next__(self) -> dict:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def open(self) -> None:
        raise NotImplementedError()

    def next_vectorized(self) -> List[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def get_description(self) -> str:
        return str(self)

    def get_structure(self) -> tuple[str, List] | str:
        return f"{id(self)}:{self.get_description()}"


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

class Criteria:
    def __init__(self, left, right):
        self.left: Criteria | Column | Constant = left
        self.right: Criteria | Column | Constant = right

    def eval(self, record) -> bool:
        raise NotImplemented()

class Negation(Criteria):
    def __init__(self, criteria: Criteria) -> None:
        super().__init__(criteria, None)

    def eval(self, record) -> bool:
        return not self.left.eval(record)

    def __str__(self):
        return f"¬{self.left}"


class ConjunctiveCriteria(Criteria):
    def __init__(self, left: Criteria, right: Criteria):
        super().__init__(left, right)

    def eval(self, record) -> bool:
        return self.left.eval(record) and self.right.eval(record)

    def __str__(self):
        return f"({self.left})∧({self.right})"


class DisjunctiveCriteria(Criteria):
    def __init__(self, left: Criteria, right: Criteria):
        super().__init__(left, right)

    def eval(self, record) -> bool:
        return self.left.eval(record) or self.right.eval(record)

    def __str__(self):
        left_str = str(self.left)
        right_srt = str(self.right)
        if len(left_str) + len(right_srt) > 30:
            return f"({left_str})∨\n({right_srt})"
        else:
            return f"({left_str})∨({right_srt})"


class HardEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__(left, right)

    def eval(self, t) -> bool:
        return self.left.get(t) == self.right.get(t)

    def __str__(self):
        return f"{self.left} = {self.right}"


class SoftEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant,
                 embedding_model: EmbeddingModel, distance: Measure = CosineSimilarity(), threshold: float = 0.9):
        super().__init__(left, right)
        self.embedding_model: EmbeddingModel = embedding_model
        self.distance: Measure = distance
        self.threshold: float = threshold

    def eval(self, t) -> bool:
        if self.left.get(t) is None or self.right.get(t) is None:
            return False

        embeddings = self.embedding_model.embedd([self.left.get(t), self.right.get(t)])
        return self.distance(embeddings[0], embeddings[1], self.threshold)

    def __str__(self):
        return f"{self.left} ≈ {self.right}"
