from typing import List

from operators import Operator, ColumnNotFoundException


class Project(Operator):
    WILDCARD: str = '*'

    def __init__(self, child_operator: Operator, columns: [List[str], WILDCARD]) -> None:
        self.child_operator: Operator = child_operator

        if columns == Project.WILDCARD:
            self.columns: List[str] = child_operator.columns
        else:
            self.columns: List[str] = []
            for column in columns:
                if column in child_operator.columns:
                    self.columns.append(column)
                else:
                    # TODO: Find columns through LLM
                    raise ColumnNotFoundException(column, child_operator.columns)

        super().__init__(self.child_operator.name, self.columns, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        return self._remap_dict(next(self.child_operator))

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        self.child_operator.open()

    def next_vectorized(self) -> List[dict]:
        data = self.child_operator.next_vectorized()
        return None if data is None else list(map(self._remap_dict, data))

    def close(self) -> None:
        self.child_operator.close()

    def get_description(self) -> str:
        return f"π_{{{', '.join(self.columns)}}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]

    def _remap_dict(self, r):
        return {col: r[col] for col in self.columns}