from typing import List

from operators import Operator


class Project(Operator):
    def __init__(self, parent_left: Operator, columns: List[str]):
        super().__init__()
        self.parent_left: Operator = parent_left
        self.columns: List[str] = columns

