import logging

from abc import ABC, abstractmethod
from typing import Literal, Callable

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
        right (Column | Constant | list[str] | None): Right input (None -> serialize Entire Record)
        em (EmbeddingModel): The model to embed a serialized record
        sv (SemanticValidationModel): The model which performs a semantic validation,
        method (threshold, zero-shot-prompting, both): Rely only on threshold/ Zero-/Few-Shot Prompting/ combination
        serialize (Callable[[dict], str]): How to convert the record to a string
        measure (Measure): The similarity/ distance function is applied to the embeddings
        threshold (float): Threshold for the similarity between both embeddings
        zfs_prompt_template (str): Format string, which is formatted to the input prompt for the SemanticValidationModel
        zfs_system_prompt (str): System prompt which is used for the SemanticValidationModel


        measure (Measure): CosineSimilarity / L2 Distance/ ...
        threshold (float)
    """
    # https://arxiv.org/pdf/2310.11244
    # ZFS_PROMPT = "Do the two entity descriptions match?\nEntity 1: {}\nEntity 2: {}"
    ZFS_PROMPT = "Do the two entity descriptions refer to the same real-world entity?\nEntity 1: {}\nEntity 2: {}"

    @staticmethod
    def default_serialization_zero_shot_prompting(x: dict) -> str:
        return ', '.join([str(v) for k, v in x.items() if v is not None])

    def __init__(self,
                 left: Column | Constant | list[str] | None,
                 right: Column | Constant | list[str] | None,
                 em: EmbeddingModel, sv: SemanticValidationModel,
                 method: Literal['threshold', 'zero-few-shot', 'both'] = 'both',
                 serialize: Callable[[dict], str] = default_serialization_zero_shot_prompting,
                 measure: Measure = CosineSimilarity(), threshold: float = 0.9,
                 zfs_prompt_template: str = ZFS_PROMPT, zfs_system_prompt: str = None):

        assert method in ('threshold', 'zero-few-shot', 'both')
        self.method = method

        if self.method == 'threshold' or self.method == 'both':
            assert em is not None and threshold is not None and measure is not None
            self.em: EmbeddingModel = em
            self.threshold: float = threshold
            self.measure: Measure = measure

        if self.method == 'zero-few-shot' or self.method == 'both':
            assert sv is not None and zfs_prompt_template is not None
            self.sv: SemanticValidationModel = sv
            self.zfs_prompt_template: str = zfs_prompt_template
            self.zfs_system_prompt: str = zfs_system_prompt

        if isinstance(left, list) or left is None or isinstance(right, list) or right is None:
            assert serialize is not None
            self.serialize = serialize

        super().__init__([left, right])


    def eval(self, t) -> bool:
        left_str = self._serialize_input(t, self.crit[0])
        right_str = self._serialize_input(t, self.crit[1])

        if self.method in ('threshold', 'both') :
            # Eval Metric: If distance -> d(v1, v2) < threshold; If Similarity -> s(v1, v2) < threshold
            embeddings = self.em([str(left_str), str(right_str)])
            _, result = self.measure(embeddings[0], embeddings[1], self.threshold)
            logging.debug(f"{left_str} ≈ {right_str}: {result}")
            if not result:
                return False

        if self.method in ('zero-few-shot', 'both'):
            prompt = self.zfs_prompt_template.format(left_str, right_str)
            result = self.sv(prompt, self.zfs_system_prompt)
            logging.debug(f"✓{prompt}: {result}")
            if not result:
                return False

        return True

    def __str__(self):
        if self.crit[0] is None or self.crit[1] is None:
            return f"{'' if self.crit[0] is None else self.crit[0]}{'' if self.crit[1] is None else self.crit[1]}"

        left = ", ".join(self.crit[0]) if isinstance(self.crit[0], list) else self.crit[1]
        right = ", ".join(self.crit[1]) if isinstance(self.crit[1], list) else self.crit[1]

        return f"{left} ≈ {right}"


    def _serialize_input(self, record, target) -> str:
        if isinstance(target, Column) or isinstance(target, Constant):
            return str(target.get(record)) # Column -> record[Column], Constant -> ConstantValue

        # Serialize Entire Record (target is None) or Reduced Record (assigned columns)
        if isinstance(target, list):
            record = {x: record[x] for x in record if x in target}

        return self.serialize(record)


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
