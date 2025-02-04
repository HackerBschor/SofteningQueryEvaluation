from typing import Callable, Any

from .Operator import Operator
from ..structure import SQLColumn, SQLTable


class ColumnTransform(Operator):
    """
    Applies a transform function (fun(dict) -> dict) for all tuples
    """

    def __init__(self, child_operator: Operator, column: SQLColumn, function: Callable[[dict], Any]):
        self.child_operator: Operator = child_operator
        self.column = column
        self.function = function

        structure = []
        has_col: bool = False
        for col in self.child_operator.table.table_structure:
           if col.column_name == column.column_name:
               # Replace existing column
               structure.append(column)
               has_col = True
           else:
               structure.append(col)

        # Add new column
        if not has_col:
            structure.append(column)

        table = SQLTable(self.child_operator.table.table_schema, self.child_operator.table.table_name, structure)

        super().__init__(table, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        rec = next(self.child_operator)
        rec[self.column.column_name] = self.function(rec)
        return rec

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        self.child_operator.open()

    def next_vectorized(self) -> list[dict]:
        data: list[dict] = self.child_operator.next_vectorized()
        for rec in data:
            rec[self.column.column_name] = self.function(rec)
        return data

    def close(self) -> None:
        raise self.child_operator.close()

    def get_description(self) -> str:
        doc: str = (" " + self.function.__doc__) if self.function.__doc__ is not None else ""
        return f"𝑓_{{→{self.column.column_name}}}"

    def get_structure(self) -> tuple[str, list] | str:
        return super().get_structure(), [self.child_operator.get_structure()]
