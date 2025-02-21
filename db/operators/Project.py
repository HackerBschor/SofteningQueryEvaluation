from abc import ABC, abstractmethod
from copy import copy
from typing import List, Type, Callable, Any

import faiss
import numpy as np

from db.operators.Operator import Operator
from db.structure import SQLTable, SQLColumn, Column, Constant

from models.embedding.Model import EmbeddingModel
from models.text_generation.Model import TextGenerationModel


class MappingFunction(ABC, Callable):
    """
    Abstract Class for Mapping Functions. Returns result of a function applied to list input columns.
        => fun(input_cols) -> output_col

    Contains semantic mapping, if the user input differs from the available columns.
    So, a Mapping for input_cols = ["name", "revenue"] and a schema [name: str, total_revenue: float]
        generates the schemantic_mapping = {"name": "name", "revenue": "total_revenue"}
    => The function input for the tuple {"name": "Robert", "total_revenue": 123.4} is fun("Robert", 123.4)

    Attributes:
        name (str): Mapping name
        input_cols (list[str]): Input columns for the function fun
        output_col (str): How the resulting column is called
        output_type (str | None): Type of the resulting column. E.g. int for math operation
        fun (Callable): Function applied to input_cols
    """
    def __init__(self, name: str | None, input_cols: list[str], output_col: str, output_type: str | None,
                 fun: Callable):
        self.name: str | None = name
        self.input_cols = input_cols
        self.output_col = output_col
        self.output_type = output_type
        self.fun = fun
        self.semantic_mapping: dict = {}

    def __call__(self, record) -> Any:
        # Reduces the record to the input cols. Use it as input for fun
        return self.fun(*(record[self.semantic_mapping[x]] for x in self.input_cols))

    def format_column(self):
        # Indicate if the user stated a correct tuple or if a semantic mapping exists
        # E.g. "name, total_revenue≈>revenue"
        cols = [c if self.semantic_mapping[c] == c else f"{self.semantic_mapping[c]}≈>{c}" for c in self.input_cols]
        return ", ".join(cols)

    @abstractmethod
    def __str__(self) -> str:
        return f"{self.name}({self.format_column()})→{self.output_col}"


class _Identity(MappingFunction):
    """ Dummy for the projection of a column (SELECT col)"""
    def __init__(self, input_name: str):
        super().__init__(None, [input_name], input_name, None, lambda r: r)

    def __str__(self) -> str:
        return self.format_column()


class _Rename(MappingFunction):
    """ Dummy for the rename of a column (SELECT col AS col_new)"""
    def __init__(self, input_name: str, output_name: str):
        super().__init__(None, [input_name], output_name, None, lambda r: r)

    def __str__(self) -> str:
        return f"{self.format_column()}→{self.output_col}"


class MathOperation(MappingFunction):
    """
    Mapping for mathematical operation (SELECT col + 1 AS col_new/ col1 * col2 AS mul)

    Example:
        MathOperation(col1, *, 2, "mul")
        for record {col1: 10} returns record {"mul": 20}

    Attributes:
        name (str): Mapping name
        left (Column | Constant): Input column/ constant
        operation (+,-,/,*): Math. operator
        right (Column | Constant): Input column/ constant
        output_col (str): How the resulting column is called
    """

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

        # Creates functions that are applied to 2 columns (col1+col2), 2 constants (2+3), column & constant (col+2)
        if isinstance(left, Column) and isinstance(right, Column):
            output_name = output_name if output_name is not None else f"{left.name}{operation}{right.name}"
            super().__init__(None, [left.name, right.name], output_name, "Integer", fun)

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
    """
    Mapping for string concat operation (SELECT col1~'-'~col2)

    Example:
        StringConcat([Column(col1), '-', Column(col1)], output_name="str", delimiter=" ")
        for record {col1: "a", col2: "b"} returns record {"str": "a - b"}

    Attributes:
        name (str): Mapping name
        input_cols (list[str]): Input columns for the function fun
        output_col (str): How the resulting column is called
        output_type (str | None): Type of the resulting column. E.g. int for math operation
        fun (Callable): Function applied to input_cols
    """

    def __init__(self, input_cols: list[str | Column], output_name: str = "str_concat", delimiter=""):
        input_cols_str = [s.name for s in input_cols if isinstance(s, Column)]

        # template for __str__  Example: (StringConcat([Column(col1), '-', Column(col1)]) -> "col1-col2)
        self.template = delimiter.join([s if isinstance(s, str) else f"{{{s}}}" for s in input_cols])

        def join(*args):
            # format_string will be filled with the tuple values
            # Example StringConcat([Column(col1), '-', Column(col1)]) -> "{}-{}".format(record["col1"], record["col1"])
            format_string = delimiter.join([s if isinstance(s, str) else "{}" for s in input_cols])
            return format_string.format(*args)

        super().__init__("str_concat", input_cols_str, output_name, "String", join)

    def __str__(self):
        return f'"{self.template}"→{self.output_col}'


class TextGeneration(MappingFunction):
    """
    Applies a LLM to a prompt template, which is filled with the input data from the record

    Example:
        StringConcat(["name", "description"], output_name="gen", prompt_template="Summarize this text for {}: "{}")
            instructs an LLM with the prompt "Summarize this text for {name}: "{description}" and returns the record
            with "gen" as the answer

    Attributes:
        input_cols (list[str]): Input columns for the function fun
        output_col (str): How the resulting column is called
        tgm (TextGenerationModel): The model that is instructed wit the prompt
        prompt_template (str): Format String, that is filled with te record
        max_new_tokens (int), temperature (float): input for hte tgm
    """

    def __init__(self, input_cols: list[str], output_col: str,
                 tgm: TextGenerationModel, prompt_template: str,
                 max_new_tokens: int = 100, temperature: float = 0.7):
        self.tgm = tgm
        self.prompt_template = prompt_template
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature

        super().__init__(None, input_cols, output_col, "String", self.generate_text)

    def generate_text(self, *args):
        # 'Summarize this text for {}: "{}"' with input_cols ["a", "b"] and record {"b": 2, "a": 1}
        # -> Summarize this text for 1: "2"
        prompt = self.prompt_template.format(*args)
        return self.tgm(prompt, None, self.max_new_tokens, self.temperature)

    def __str__(self) -> str:
        cols = [f"<{x}>" for x in self.input_cols]
        return f'𝒯("{self.prompt_template.format(*cols)}")→{self.output_col}'


class CustomMapping(MappingFunction):
    """
    Can be defined by the users, so any function is applied on the records input cols
    """
    def __init__(self, input_cols: list[str], output_col: str, output_type: str, fun: Callable):
        super().__init__(None, input_cols, output_col, output_type, fun)

    def __str__(self) -> str:
        cols = [f"<{x}>" for x in self.input_cols]
        return f'CUSTOM("{', '.join(self.input_cols)}")→{self.output_col}'


class Project(Operator):
    """
    Applies Mappings to record form child operator and returns mapped record.

    E.g. Mappings: ["a", ("b", "b_new"), MathOperation("a", "+", "b", "sum), "c"]
        Can be compared to "SELECT a, b AS b_new, a+b AS sum, c"

    For all columns used by the mappings, a semantic-search for the correct column is performed
    E.g. Mappings: ["title", ("title_en", "original_title"), MathOperation("revenue", "/", 1000, "revenue_million")
        for a schema ["movie_title", "title_english", "revenue_B"] can be evaluated as
        ["movie_title", ("title_english", "original_title"), MathOperation("revenue_B", "/", 1000, "revenue_million")

    Attributes:
        child_operator (Operator): The operator that generates the records for filtering
        columns (list[str | tuple[str, str] | MappingFunction]): The mapping functions which will be applied to records
        em (EmbeddingModel): The model to embedd a column name
        vs (faiss.IndexFlat): Vector Store for semantic search
        threshold (float): Threshold for the similarity between stated column name embedding and available column name embedding
    """
    def __init__(self, child_operator: Operator, columns: list[str | tuple[str, str] | MappingFunction],
                 em: EmbeddingModel, vs: Type[faiss.IndexFlat] = faiss.IndexFlatIP, threshold: float = 0.1) -> None:

        self.child_operator: Operator = child_operator

        # Convert the input to list of mappings (str -> Identity, (str, str) -> rename, Mapping -> Mapping
        self.mappings = [Project._convert_input_column(column) for column in columns]

        # Columns provided by the child_operator
        available_columns = {col.column_name: col for col in child_operator.table.table_structure}

        # If the (user-declared) column not present in the available in the schema -> schema bindings
        columns = {"use": set(), "search": set()}
        for mapping in self.mappings:
            for column in mapping.input_cols:
                # adds column used by the mapping to "use", if its in available_columns, otherwise to "search"
                columns["use" if column in available_columns else "search"].add(column)

        # column_mappings: dict[declared columns by user -> available columns]
        # The semantic_column_mappings is used to retrieve the correct column from the record
        # E.g. schema: ["name", "movie_title"] and user declared ["name", "title"]:
        #   Mapping = {"name": "name", "movie_title": "title"}
        self.semantic_column_mappings = {x: x for x in columns["use"]} # all available columns are mapped to itself
        if len(columns["search"]) > 0:
            # For the rest search in data
            self._semantic_column_search(available_columns, columns["search"], em, vs, threshold)

        # Create new Schema for parent operator from mapping outputs
        schema_columns = []
        for mapping in self.mappings:
            mapping.semantic_mapping = self.semantic_column_mappings

            if isinstance(mapping, _Identity) or isinstance(mapping, _Rename):
                # When Identity/ Rename -> Copy & rename column from schema of child operator
                reference_column = available_columns[self.semantic_column_mappings[mapping.input_cols[0]]]
                schema_column = copy(reference_column)
                schema_column.column_name = mapping.output_col
            else:
                # When Mapping -> Create new column
                schema_column = SQLColumn(mapping.output_col, mapping.output_type)

            schema_columns.append(schema_column)

        t = SQLTable(child_operator.table.table_schema, child_operator.table.table_name, schema_columns)
        super().__init__(t, self.child_operator.num_tuples)

    def __next__(self) -> dict:
        # Applies mappings, return mapped record
        return self._remap_dict(next(self.child_operator))

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> Operator:
        self.child_operator.open()
        return self

    def next_vectorized(self) -> List[dict]:
        # Applies mappings to *N* records, return list fo mapped record
        data = self.child_operator.next_vectorized()
        return None if data is None else list(map(self._remap_dict, data))

    def close(self) -> None:
        self.child_operator.close()

    def get_description(self) -> str:
        return f"π_{{{', '.join(map(str, self.mappings))}}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]

    def _remap_dict(self, r):
        # Applies mappings to record
        return {mapping.output_col: mapping(r) for mapping in self.mappings}

    @staticmethod
    def _convert_input_column(column: [str | tuple[str, str] | MappingFunction]) -> MappingFunction:
        # Convert the input to mapping (str -> Identity, (str, str) -> rename, Mapping -> Mapping
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
            assert len(indices) > 0, f"No column available for input {col}, available columns: {', '.join([k for k in available_columns])}"

            if vector_store.metric_type == faiss.METRIC_L2:
                closest_column_index = indices[np.argmin(distances)]
            else:
                closest_column_index = indices[np.argmax(distances)]

            self.semantic_column_mappings[col] = available_column_names[closest_column_index]
