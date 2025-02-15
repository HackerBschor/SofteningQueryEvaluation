from typing import List

from .Operator import Operator


class Union(Operator):
    """
    Filters tuples according to a provided criteria
    """

    def __init__(self, ops: list[Operator]) -> None:
        assert len(ops) > 0

        self.ops = ops
        self.structure_mapping = {}

        for op in ops:
            assert len(ops[0].table.table_structure) == len(op.table.table_structure), \
                f"{op.table.table_structure} is not equal to {len(op.table.table_structure)} "

            for col1, col2 in zip(ops[0].table.table_structure, op.table.table_structure):
                self.structure_mapping[col2.column_name] = col1.column_name

        self.iter_tuples = None
        self.iter_operators = None

        super().__init__(self.ops[0].table, self.ops[0].num_tuples)

    def __next__(self) -> dict:
        while True:
            try:
                r = next(self.iter_tuples)
                r = {self.structure_mapping[k]: v for k,v in r.items()}
                return r
            except StopIteration:
                op = next(self.iter_operators)
                op.open()
                self.iter_tuples = iter(op)


    def __str__(self) -> str:
        return f"{self.get_description()} ({', '.join(map(str, self.ops))})"

    def open(self) -> None:
        self.iter_operators = iter(self.ops)
        op = next(self.iter_operators)
        op.open()
        self.iter_tuples = iter(op)

    def next_vectorized(self) -> List[dict]:
        raise NotImplementedError

    def close(self) -> None:
        self.iter_operators = None
        self.iter_tuples = None

        for op in self.ops:
            op.close()

    def get_description(self) -> str:
        return f"∪"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [op.get_structure() for op in self.ops]
