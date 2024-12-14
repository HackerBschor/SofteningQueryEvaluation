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

        super().__init__(self.child_operator.name, columns)

    def __str__(self):
        return f"π({', '.join(self.columns)})"

    def __next__(self) -> dict:
        r: dict = next(self.child_operator)
        return {col: r[col] for col in self.columns}

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]