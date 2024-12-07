from operators import Operator


class Scan(Operator):
    def __init__(self, parent: Operator, key):
        super().__init__()
        self.parent: Operator = parent
        self.key = key
