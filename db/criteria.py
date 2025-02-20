import logging

from abc import ABC, abstractmethod

from utils import Measure, CosineSimilarity

from models.semantic_validation.Model import SemanticValidationModel
from models.embedding.Model import EmbeddingModel

from db.structure import Column, Constant


class Criteria(ABC):
    """
    Abstract base class for criteria
    """
    def __init__(self, crit):
        self.crit = crit

    @abstractmethod
    def eval(self, record) -> bool:
        raise NotImplemented()

class Negation(Criteria):
    """
    Returns the negation of an input criterion
    """
    def __init__(self, crit: Criteria) -> None:
        super().__init__(crit)

    def eval(self, record) -> bool:
        return not self.crit.eval(record)

    def __str__(self):
        return f"¬{self.crit}"


class ConjunctiveCriteria(Criteria):
    """
    Evaluates input criteria. Returns true if ALL are fulfilled
    """
    def __init__(self, crit: list[Criteria]):
        super().__init__(crit)

    def eval(self, record) -> bool:
        return all([crit.eval(record) for crit in self.crit])

    def __str__(self):
        return f"({'∧'.join(map(lambda x: f'({x})', self.crit))}"


class DisjunctiveCriteria(Criteria):
    """
    Evaluates input criteria. Returns true if AT LEAST ONE is fulfilled
    """
    def __init__(self, crit: list[Criteria]):
        super().__init__(crit)

    def eval(self, record) -> bool:
        return any([crit.eval(record) for crit in self.crit])

    def __str__(self):
        return f"({'∨'.join(map(lambda x: f'({x})', self.crit))}"


class HardEqual(Criteria):
    """
    Evaluates if two inputs (Column Value or constant) are equal. E.g. name = 'Robert' or 15.0 = price
    """
    def __init__(self, left: Column | Constant, right: Column | Constant):
        super().__init__([left, right])

    def eval(self, t) -> bool:
        return self.crit[0].get(t) == self.crit[1].get(t)

    def __str__(self):
        return f"{self.crit[0]} = {self.crit[1]}"


class IsNull(Criteria):
    """
    Evaluates if an input is null
    """
    def __init__(self, left: Column | Constant):
        super().__init__([left, None])

    def eval(self, t) -> bool:
        return self.crit[0].get(t) is None

    def __str__(self):
        return f"{self.crit[0]} is null"


class IsNotNull(Criteria):
    """
    Evaluates if an input is NOT null
    """
    def __init__(self, left: Column | Constant):
        super().__init__([left, None])

    def eval(self, t) -> bool:
        return self.crit[0].get(t) is not None

    def __str__(self):
        return f"{self.crit[0]} is not null"


class FuzzyEqual:
    # TODO: Implement for comparison
    pass


class SoftEqual(Criteria):
    """
    Evaluation Left ≈ Right:
        Evaluates if the distance (or similarity score) of the embedded of two
        serialized inputs (Constant, Column Value, list of Column Values, Entire Record)
        is below (or above for similarly ) a threshold

    E.g. movie_title ≈ 'Menu', [movie_title, description] ≈ 'toys that come to life'

    Attributes:
        left (Column | Constant | list[str] | None): Left input (None -> serialize Entire Record)
        right (Column | Constant | list[str] | None):  Right input (None -> serialize Entire Record)
        em (EmbeddingModel): The model to embedd a serialized database table
        measure (Measure): CosineSimilarity / L2 Distance/ ...
        threshold (float)
    """
    def __init__(self,
                 left: Column | Constant | list[str] | None,
                 right: Column | Constant | list[str] | None,
                 em: EmbeddingModel, measure: Measure = CosineSimilarity(), threshold: float = 0.9):
        super().__init__([left, right])
        self.em: EmbeddingModel = em
        self.measure: Measure = measure
        self.threshold: float = threshold

    def eval(self, t) -> bool:
        emb_str_left = self._serialize_input(t, self.crit[0])
        emb_str_right = self._serialize_input(t, self.crit[1])

        # Eval Metric: If distance -> d(v1, v2) < threshold; If Similarity -> s(v1, v2) < threshold
        embeddings = self.em([str(emb_str_left), str(emb_str_right)])
        result = self.measure(embeddings[0], embeddings[1], self.threshold)
        logging.debug(f"{emb_str_left} ≈ {emb_str_right}: {result}")
        return result

    def __str__(self):
        if self.crit[0] is None or self.crit[1] is None:
            return f"{'' if self.crit[0] is None else self.crit[0]}{'' if self.crit[1] is None else self.crit[1]}"

        left = ", ".join(self.crit[0]) if isinstance(self.crit[0], list) else self.crit[1]
        right = ", ".join(self.crit[1]) if isinstance(self.crit[1], list) else self.crit[1]

        return f"{left} ≈ {right}"


    @staticmethod
    def _serialize_input(record, target):
        if target is None:
            return str(record) # Serialize Entire Record
        elif isinstance(target, list):
            return str({x: record[x] for x in record if x in target}) # Serialize reduced record (assigned columns)
        else:
            return target.get(record) # Column -> record[Column], Constant -> ConstantValue


class SoftValidate(Criteria):
    """
    Evaluation ✓(template):
        Instructs an LLM with the prompt generated from template and the tuple. The template must be a format string.
        Evaluate if the response of the LLM is yes/ no.


    E.g. ✓("Is the movie {movie_title} is about toys that come to life?") for the record {"movie_title": "Toy Story"}
        would evaluate to ✓("Is the movie Toy Story is about toys that come to life?") -> True

    Attributes:
        template (str): Format String
        sv (SemanticValidationModel): The generative model which will be instructed
    """
    def __init__(self, template: str, sv: SemanticValidationModel):
        super().__init__(None)
        self.template: str | None = template
        self.sv: SemanticValidationModel = sv


    def eval(self, t) -> bool:
        prompt = self.template.format(**t)
        result = self.sv(prompt)
        logging.debug(f"{prompt}: {result}")
        return result

    def __str__(self):
        return f"✓_{{{self.template}}}"
