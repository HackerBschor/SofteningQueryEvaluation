from copy import copy
from typing import List, Type, Callable, Any

import faiss
import numpy as np

from db.operators.Operator import Operator
from db.structure import SQLTable, SQLColumn, Column, Constant

from models.embedding.Model import EmbeddingModel
from models.text_generation.Model import TextGenerationModel


class MappingFunction:
    def __init__(self, name: str | None, input_cols: list[str], output_col: str, output_type: str | None,
                 fun: Callable):
        self.name: str | None = name
        self.input_cols = input_cols
        self.output_col = output_col
        self.output_type = output_type
        self.fun = fun
        self.semantic_mapping: dict = {}

    def __call__(self, record) -> Any:
        return self.fun(*(record[self.semantic_mapping[x]] for x in self.input_cols))

    def format_column(self):
        cols = [c if self.semantic_mapping[c] == c else f"{self.semantic_mapping[c]}≈>{c}" for c in self.input_cols]
        return ", ".join(cols)

    def __str__(self) -> str:
        return f"{self.name}({self.format_column()})→{self.output_col}"


class _Identity(MappingFunction):
    def __init__(self, input_name: str):
        super().__init__(None, [input_name], input_name, None, lambda r: r)

    def __str__(self) -> str:
        return self.format_column()


class _Rename(MappingFunction):
    def __init__(self, input_name: str, output_name: str):
        super().__init__(None, [input_name], output_name, None, lambda r: r)

    def __str__(self) -> str:
        return f"{self.format_column()}→{self.output_col}"


class MathOperation(MappingFunction):
    Operations = ["+", "-", "*", "/"]

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def sub(x, y):
        return x - y

    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def div(x, y):
        return x / y

    def __init__(self, left: Column | Constant, operation: Operations, right: Column | Constant,
                 output_name: str = None):
        self.operation = operation
        self.left = left
        self.right = right
        fun = self.get_operation(operation)

        if isinstance(left, Column) and isinstance(right, Column):
            input_cols = [left.name, right.name]
            output_name = output_name if output_name is not None else operation.join(input_cols)
            super().__init__(None, input_cols, output_name, "Integer", fun)

        elif isinstance(left, Constant) and isinstance(right, Constant):
            result = fun(left.value, right.value)
            output_name = output_name if output_name is not None else str(result)
            super().__init__(None, [], output_name, "Integer", lambda: result)

        else:
            if isinstance(left, Column) and isinstance(right, Constant):
                output_name = output_name if output_name is not None else f"{left.name}{operation}{right.value}"
                super().__init__(None, [left.name], output_name, "Integer", lambda x: fun(x, right.value))
            else:
                output_name = output_name if output_name is not None else f"{left.value}{operation}{right.name}"
                super().__init__(None, [right.name], output_name, "Integer", lambda x: fun(left.value, x))

    @staticmethod
    def get_operation(op: Operations):
        if op == "+":
            return MathOperation.add
        elif op == "-":
            return MathOperation.sub
        elif op == "*":
            return MathOperation.mul
        else:
            return MathOperation.div

    def __str__(self) -> str:
        return f"{self.left}{self.operation}{self.right}"


class StringConcat(MappingFunction):
    def __init__(self, input_cols: list[str | Column], output_name: str = "str_concat", delimiter=""):
        input_cols_str = [s.name for s in input_cols if isinstance(s, Column)]
        self.template = delimiter.join([s if isinstance(s, str) else f"{{{s}}}" for s in input_cols])

        def join(*args):
            return delimiter.join([s if isinstance(s, str) else "{}" for s in input_cols]).format(*args)

        super().__init__("str_concat", input_cols_str, output_name, "String", join)

    def __str__(self):
        return f'"{self.template}"→{self.output_col}'


class TextGeneration(MappingFunction):
    def __init__(self, input_cols: list[str], output_col: str,
                 tgm: TextGenerationModel, prompt_template: str,
                 system_prompt: str | None = None, max_new_tokens: int = 100, temperature: float = 0.7):
        self.tgm = tgm
        self.prompt_template = prompt_template
        self.system_prompt: str = system_prompt
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature

        super().__init__(None, input_cols, output_col, "String", self.generate_text)

    def generate_text(self, *args):
        prompt = self.prompt_template.format(*args)
        return self.tgm(prompt, self.system_prompt, self.max_new_tokens, self.temperature)

    def __str__(self) -> str:
        cols = [f"<{x}>" for x in self.input_cols]
        return f'𝒯("{self.prompt_template.format(*cols)}")→{self.output_col}'


class CustomMapping(MappingFunction):
    def __init__(self, input_cols: list[str], output_col: str, output_type: str, fun: Callable):
        super().__init__(None, input_cols, output_col, output_type, fun)

    def __str__(self) -> str:
        cols = [f"<{x}>" for x in self.input_cols]
        return f'CUSTOM("{', '.join(self.input_cols)}")→{self.output_col}'


class Project(Operator):
    def __init__(self, child_operator: Operator, columns: list[str | tuple[str, str] | MappingFunction],
                 em: EmbeddingModel, vs: Type[faiss.IndexFlat] = faiss.IndexFlatIP, threshold: float = 0.8) -> None:

        self.child_operator: Operator = child_operator
        self.mappings = [Project._convert_input_column(column) for column in columns]

        available_columns = {col.column_name: col for col in child_operator.table.table_structure}

        columns = {"use": set(), "search": set()}
        for mapping in self.mappings:
            for column in mapping.input_cols:
                columns["use" if column in available_columns else "search"].add(column)

        # column_mappings: dict[declared columns -> available columns]
        self.semantic_column_mappings = {x: x for x in columns["use"]}
        if len(columns["search"]) > 0:
            self._semantic_column_search(available_columns, columns["search"], em, vs, threshold)

        schema_columns = []
        for mapping in self.mappings:
            mapping.semantic_mapping = self.semantic_column_mappings

            if isinstance(mapping, _Identity) or isinstance(mapping, _Rename):
                reference_column = available_columns[self.semantic_column_mappings[mapping.input_cols[0]]]
                schema_column = copy(reference_column)
                schema_column.column_name = mapping.output_col
            else:
                schema_column = SQLColumn(mapping.output_col, mapping.output_type)

            schema_columns.append(schema_column)

        t = SQLTable(child_operator.table.table_schema, child_operator.table.table_name, schema_columns)
        super().__init__(t, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        return self._remap_dict(next(self.child_operator))

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        self.child_operator.open()

    def next_vectorized(self) -> List[dict]:
        data = self.child_operator.next_vectorized()
        return None if data is None else list(map(self._remap_dict, data))

    def close(self) -> None:
        self.child_operator.close()

    def get_description(self) -> str:
        return f"π_{{{', '.join(map(str, self.mappings))}}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]

    def _remap_dict(self, r):
        return {mapping.output_col: mapping(r) for mapping in self.mappings}

    @staticmethod
    def _convert_input_column(column: [str | tuple[str, str] | MappingFunction]) -> MappingFunction:
        if isinstance(column, str):
            return _Identity(column)
        elif isinstance(column, tuple):
            return _Rename(column[0], column[1])
        elif isinstance(column, MappingFunction):
            return column
        else:
            raise TypeError(f'Unsupported column type: {type(column)}')

    def _semantic_column_search(self, available_columns: dict[str, SQLColumn], search_columns: set[str],
                                em: EmbeddingModel, vs: Type[faiss.IndexFlat], threshold: float):
        vector_store = vs(em.get_embedding_size())
        available_column_names, search_column_names = list(available_columns), list(search_columns)
        embeddings = em(available_column_names + search_column_names)

        # noinspection PyArgumentList
        vector_store.add(embeddings[:len(available_columns)])

        for i, (col, emb) in enumerate(zip(search_columns, embeddings[len(available_columns):])):
            # noinspection PyArgumentList
            _, distances, indices = vector_store.range_search(x=np.array([emb]), thresh=threshold)
            assert len(indices) > 0, f"No column available for input {col}, available columns: {available_columns}"

            if vector_store.metric_type == faiss.METRIC_L2:
                closest_column_index = indices[np.argmin(distances)]
            else:
                closest_column_index = indices[np.argmax(distances)]

            self.semantic_column_mappings[col] = available_column_names[closest_column_index]
