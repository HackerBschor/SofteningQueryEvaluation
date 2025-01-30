from typing import List

from models.text_generation.Model import TextGenerationModel
from .Operator import Operator
from ..criteria import Criteria
from ..structure import SQLColumn


class TextGeneration(Operator):
    """
    Instruct an TextGenerationLLM to generate text
    """

    def __init__(self, child_operator: Operator, tgm: TextGenerationModel,
                 prompt_template: str, system_prompt: str | None = None, max_new_tokens: int = 100,
                 temperature: float = 0.7, new_column_name:  str = "response") -> None:

        self.child_operator: Operator = child_operator
        self.tgm: TextGenerationModel = tgm
        self.prompt_template: str = prompt_template
        self.system_prompt: str = system_prompt
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.new_column_name: str = new_column_name
        assert new_column_name not in [col.column_name for col in self.child_operator.table.table_structure], "Column name already exists"
        self.child_operator.table.table_structure.append(SQLColumn(self.new_column_name, "Text"))

        super().__init__(self.child_operator.table, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        record = next(self.child_operator)
        prompt = self.prompt_template.format(**record)
        record[self.new_column_name] = self.tgm(prompt, self.system_prompt, self.max_new_tokens, self.temperature)
        return record


    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        self.child_operator.open()

    def next_vectorized(self) -> List[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.child_operator.close()

    def get_description(self) -> str:
        return f"𝒯_{{'{ self.prompt_template }'}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]