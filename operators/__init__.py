import re

from typing import Iterator, List, Any

from utils import Measure, CosineSimilarity
from models.embedding.Model import EmbeddingModel


class ColumnNotFoundException(Exception):
    def __init__(self, column: str, columns: List[str]) -> None:
        self.column: str = column
        self.columns: List[str] = columns

    def __str__(self) -> str:
        return f"Column '{self.column}' not found in parent columns {self.columns}"

########################################
######## SQL Structure
########################################

class SQLColumn:
    PATTERN_FOREIGN_KEY = re.compile(r'FOREIGN_KEY\((.+)\.(.+)\.(.+)\)')

    def __init__(self, column_name, column_type, primary_key: bool = False, foreign_key: list[tuple[str, str, str]] = ()):
        self.column_name = column_name
        self.column_type = column_type
        self.primary_key = primary_key
        self.foreign_key: list[tuple[str, str, str]] = foreign_key

    @staticmethod
    def parse(descr) -> 'SQLColumn':
        descr_enc = descr.split(":")
        if len(descr_enc) < 2:
            raise Exception(f"Invalid column description: {descr}")
        elif len(descr_enc) == 2:
            return SQLColumn(descr_enc[0], descr_enc[1])
        else:
            col = SQLColumn(descr_enc[0], descr_enc[1])

            for attr in descr_enc[2:]:
                if attr == "PRIMARY_KEY":
                    col.primary_key = True
                if attr.startswith("FOREIGN_KEY"):
                    match = SQLColumn.PATTERN_FOREIGN_KEY.match(attr)
                    if match:
                        (s, t, c) = match.groups()
                        col.foreign_key.append((s, t, c))

            return col

    def __str__(self) -> str:
        res = self.column_name + "(" + self.column_type
        res += ", PRIMARY_KEY" if self.primary_key else ""
        for fk in self.foreign_key:
            res += f", FOREIGN_KYE({fk[0]}.{fk[1]}.{fk[2]})"
        return res + ")"


class SQLTable:
    def __init__(self, table_schema: str | None, table_name: str, table_structure: str | list[SQLColumn]):
        self.table_schema: str | None = table_schema
        self.table_name: str = table_name
        if isinstance(table_structure, str):
            self.table_structure: list[SQLColumn] = self._parse_structure(table_structure)
        else:
            self.table_structure = table_structure

    def __str__(self):
        if self.table_schema:
            return f"{self.table_schema}.{self.table_name}: {', '.join(map(str, self.table_structure))}"
        else:
            return f"{self.table_name}: {', '.join(map(str, self.table_structure))}"

    @staticmethod
    def _parse_structure(table_structure: str) -> list[SQLColumn]:
        return [SQLColumn.parse(col) for col in table_structure.split(", ")]


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

    def next_vectorized(self) -> List[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def get_description(self) -> str:
        return str(self)

    def get_structure(self) -> tuple[str, List] | str:
        return f"{id(self)}:{self.get_description()}"

########################################
######## Column / Constant
########################################

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
