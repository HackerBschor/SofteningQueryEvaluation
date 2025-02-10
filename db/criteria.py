import logging

from models.semantic_validation.Model import SemanticValidationModel
from utils import Measure, CosineSimilarity
from models.embedding.Model import EmbeddingModel
from .structure import Column, Constant


class Criteria:
    def __init__(self, crit):
        self.crit = crit

    def eval(self, record) -> bool:
        raise NotImplemented()

class Negation(Criteria):
    def __init__(self, crit: Criteria) -> None:
        super().__init__(crit)

    def eval(self, record) -> bool:
        return not self.crit.eval(record)

    def __str__(self):
        return f"¬{self.crit}"


class ConjunctiveCriteria(Criteria):
    def __init__(self, crit: list[Criteria]):
        super().__init__(crit)

    def eval(self, record) -> bool:
        return all([crit.eval(record) for crit in self.crit])

    def __str__(self):
        return f"({'∧'.join(map(lambda x: f'({x})', self.crit))}"


class DisjunctiveCriteria(Criteria):
    def __init__(self, crit: list[Criteria]):
        super().__init__(crit)

    def eval(self, record) -> bool:
        return any([crit.eval(record) for crit in self.crit])

    def __str__(self):
        return f"({'∨'.join(map(lambda x: f'({x})', self.crit))}"


class HardEqual(Criteria):
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__([left, right])

    def eval(self, t) -> bool:
        return self.crit[0].get(t) == self.crit[1].get(t)

    def __str__(self):
        return f"{self.crit[0]} = {self.crit[1]}"


class FuzzyEqual:
    # TODO: Implement for comparison
    raise NotImplementedError


class SoftEqual(Criteria):
    def __init__(self,
                 left: Column | Constant | list[str] | None,
                 right: Column | Constant | list[str] | None,
                 em: EmbeddingModel, distance: Measure = CosineSimilarity(), threshold: float = 0.9):
        super().__init__([left, right])
        self.em: EmbeddingModel = em
        self.distance: Measure = distance
        self.threshold: float = threshold

    def eval(self, t) -> bool:
        emb_str_left = self._get_embedding_string(t, self.crit[0])
        emb_str_right = self._get_embedding_string(t, self.crit[1])

        embeddings = self.em([emb_str_left, emb_str_right])
        result = self.distance(embeddings[0], embeddings[1], self.threshold)
        logging.debug(f"{emb_str_left} ≈ {emb_str_right}: {result}")
        return result

    def __str__(self):
        if self.crit[0] is None or self.crit[1] is None:
            return f"{'' if self.crit[0] is None else self.crit[0]}{'' if self.crit[1] is None else self.crit[1]}"

        left = ", ".join(self.crit[0]) if isinstance(self.crit[0], list) else self.crit[1]
        right = ", ".join(self.crit[1]) if isinstance(self.crit[1], list) else self.crit[1]

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
        super().__init__(None)
        self.template: str | None = template
        self.sv: SemanticValidationModel = sv
        self.full_record = full_record


    def eval(self, t) -> bool:
        prompt = self.template.format(str(t)) if self.full_record else self.template.format(**t)
        result = self.sv(prompt)
        logging.debug(f"{prompt}: {result}")
        return result

    def __str__(self):
        return f"✓_{{{self.template}}}"
