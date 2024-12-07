from operators import Operator


class Scan(Operator):
    def __init__(self, parent_left: Operator, parent_right: Operator, key):
        super().__init__()
        self.parent_left: Operator = parent_left
        self.parent_right: Operator = parent_right
        self.key = key
