import logging

from models.semantic_validation.Model import SemanticValidationModel
from utils import Measure, CosineSimilarity
from models.embedding.Model import EmbeddingModel
from .structure import Column, Constant


class Criteria:
    def __init__(self, left, right):
        self.left: Criteria | Column | Constant | list[Column]| None  = left
        self.right: Criteria | Column | Constant | list[Column] | None = right

    def eval(self, record) -> bool:
        raise NotImplemented()

class Negation(Criteria):
    def __init__(self, criteria: Criteria) -> None:
        super().__init__(criteria, None)

    def eval(self, record) -> bool:
        return not self.left.eval(record)

    def __str__(self):
        return f"¬{self.left}"


class ConjunctiveCriteria(Criteria):
    def __init__(self, left: Criteria, right: Criteria):
        super().__init__(left, right)

    def eval(self, record) -> bool:
        return self.left.eval(record) and self.right.eval(record)

    def __str__(self):
        new_line = "\n" if len(str(self.left)) + len(self.right) > 30 else ""
        return f"({self.left})∧{new_line}({self.right})"


class DisjunctiveCriteria(Criteria):
    def __init__(self, left: Criteria, right: Criteria):
        super().__init__(left, right)

    def eval(self, record) -> bool:
        return self.left.eval(record) or self.right.eval(record)

    def __str__(self):
        new_line = "\n" if len(str(self.left)) + len(self.right) > 30 else ""
        return f"({self.left})∨{new_line}({self.right})"


class HardEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__(left, right)

    def eval(self, t) -> bool:
        return self.left.get(t) == self.right.get(t)

    def __str__(self):
        return f"{self.left} = {self.right}"


class SoftEqual(Criteria):
    def __init__(self,
                 left: Column | Constant | list[str] | None,
                 right: Column | Constant | list[str] | None,
                 em: EmbeddingModel, distance: Measure = CosineSimilarity(), threshold: float = 0.9):
        super().__init__(left, right)
        self.em: EmbeddingModel = em
        self.distance: Measure = distance
        self.threshold: float = threshold

    def eval(self, t) -> bool:
        emb_str_left = self._get_embedding_string(t, self.left)
        emb_str_right = self._get_embedding_string(t, self.right)

        logging.debug(f"{emb_str_left} ≈ {emb_str_right}")

        embeddings = self.em([emb_str_left, emb_str_right])

        return self.distance(embeddings[0], embeddings[1], self.threshold)

    def __str__(self):
        if self.left is None or self.right is None:
            return f"{'' if self.left is None else self.left}{'' if self.right is None else self.right}"

        left = ", ".join(self.left) if isinstance(self.left, list) else self.left
        right = ", ".join(self.right) if isinstance(self.right, list) else self.right

        return f"{left} ≈ {right}"


    @staticmethod
    def _get_embedding_string(record, target):

        if target is None:
            return str(record)
        if isinstance(target, list):
            return str({x: record[x] for x in record if x in target})
        else:
            return target.get(record)


class SoftValidate(Criteria):
    def __init__(self, template: str, sv: SemanticValidationModel, full_record: bool = True):
        super().__init__(None, None)
        self.template: str | None = template
        self.sv: SemanticValidationModel = sv
        self.full_record = full_record


    def eval(self, t) -> bool:
        prompt = self.template.format(str(t)) if self.full_record else self.template.format(**t)
        return self.sv(prompt)

    def __str__(self):
        return f"✓_{{{self.template}}}"
