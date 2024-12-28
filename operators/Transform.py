from typing import Callable, List

from operators import Operator


class Transform(Operator):
    """ Applies a transform function (fun(dict) -> dict) for all tuples """

    def __init__(self, child_operator: Operator, function: Callable[[dict], dict]):
        self.child_operator: Operator = child_operator
        self.function: Callable[[dict], dict] = function
        super().__init__(child_operator.name, child_operator.columns, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        return self.function(next(self.child_operator))

    def get_description(self) -> str:
        doc = (" " + self.function.__doc__) if self.function.__doc__ is not None else ""
        return f"✨ {self.function.__name__}{doc} ✨"

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]