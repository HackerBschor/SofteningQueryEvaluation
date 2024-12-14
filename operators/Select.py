from typing import List

from operators import Operator, Column, Constant

from utils.Model import EmbeddingModel, GenerationModel
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class Criteria:
    def __init__(self, left, right):
        self.left: Criteria | Column | Constant = left
        self.right: Criteria | Column | Constant = right

    def eval(self, record) -> bool:
        raise NotImplemented()


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
        return f"({self.left})∨({self.right})"


class HardEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__(left, right)

    def eval(self, t) -> bool:
        return self.left.get(t) == self.right.get(t)

    def __str__(self):
        return f"{self.left} = {self.right}"


class HardUnEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__(left, right)

    def eval(self, t) -> bool:
        return self.left.get(t) != self.right.get(t)

    def __str__(self):
        return f"{self.left} ≠ {self.right}"


class SoftEqualEmbedding(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant,
                 embedding_model: EmbeddingModel, threshold: float = 0.9):
        super().__init__(left, right)
        self.embedding_model: EmbeddingModel = embedding_model
        self.threshold: float = threshold

    def eval(self, t) -> bool:
        if self.left.get(t) is None or self.right.get(t) is None:
            return False
        embedding_left = self.embedding_model.embedd(self.left.get(t))
        embedding_right = self.embedding_model.embedd(self.right.get(t))
        cs = cosine_similarity(embedding_left[0], embedding_right[0])
        print(self.left.get(t), self.right.get(t), cs)
        return cs > self.threshold

    def __str__(self):
        return f"{self.left} ≈ {self.right}"


class SoftEqualGeneration(Criteria):
    PROMPT_TEMPLATE = "Would you consider '{}' to be '{}'. Answer with 'yes' or 'no' only!"

    def __init__(self, left: Column | Constant, right: Column | Constant,
                 generation_model: GenerationModel):
        super().__init__(left, right)
        self.generation_model: GenerationModel = generation_model

    def eval(self, t) -> bool:
        if self.left.get(t) is None or self.right.get(t) is None:
            return False

        prompt = self.PROMPT_TEMPLATE.format(self.left.get(t), self.right.get(t))
        result = self.generation_model.generate(
            prompt,
            max_new_tokens=1,
            temperature=1,
            top_p=0.95,
            top_k=50,
            do_sample=True).replace(prompt, "")

        print(self.left.get(t), self.right.get(t), ": ", result)
        return False

    def __str__(self):
        return f"{self.left} ≈ {self.right}"


class Select(Operator):
    """ Filters tuples according to a provided criteria """

    def __init__(self, child_operator: Operator, criteria: Criteria):
        self.child_operator: Operator = child_operator
        self.criteria: Criteria = criteria
        super().__init__(self.child_operator.name, self.child_operator.columns)

    def __str__(self):
        return f"σ({self.criteria})"

    def __next__(self) -> dict:
        for t in self.child_operator:
            if self.criteria.eval(t):
                return t

        raise StopIteration

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]
