from typing import List

from db.operators.Operator import Operator
from db.criteria import Criteria

# TODO: Increase Performance using Lookup Tables
# # When selecting on a categorical field and |categories| << n -> build Lookup Table {Category -> bool}


class Select(Operator):
    """
    Filters tuples according to a provided criterion
    Construct criterion using the criteria in db.criteria.py

    Attributes:
        child_operator (Operator): The operator that generates the records for filtering
        criteria (Criteria): The criteria to filter on
    """

    def __init__(self, child_operator: Operator, criteria: Criteria) -> None:
        self.child_operator: Operator = child_operator
        self.criteria: Criteria = criteria
        super().__init__(self.child_operator.table, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        # Fetches a next tuple from the child operator until the criteria is fulfilled
        for t in self.child_operator:
            if self.criteria.eval(t):
                return t

        raise StopIteration

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> Operator:
        self.child_operator.open()
        return self

    def next_vectorized(self) -> List[dict]:
        data = self.child_operator.next_vectorized()
        return None if data is None else [t for t in data if self.criteria.eval(t)]  # TODO: Batch Processing

    def close(self) -> None:
        self.child_operator.close()

    def get_description(self) -> str:
        return f"σ_{{{self.criteria}}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]
