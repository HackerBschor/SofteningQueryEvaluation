import faiss
import numpy as np
import logging

from typing import Any, Type, Optional, Literal, Callable

from db.operators.Operator import Operator
from db.structure import SQLColumn, SQLTable, Column
from db.criteria import Criteria

from models.embedding.Model import EmbeddingModel
from models.semantic_validation.Model import SemanticValidationModel


class Join(Operator):
    def __init__(self, child_left: Operator, child_right: Operator, criteria: Criteria | None):
        self.child_left: Operator = child_left
        self.child_right: Operator = child_right
        self.criteria: Criteria = criteria
        self.current_tuple = None

        name = f"{self.child_left.table.table_name}+{self.child_right.table.table_name}"

        # builds columns map: a = {x, y} & b = {y, z} -> columns_map = {left: {x: x, y: a.y}, right: {z: z, y: b.y}}
        columns_left = {x.column_name for x in self.child_left.table.table_structure}
        columns_right = {x.column_name for x in self.child_right.table.table_structure}
        columns_intersect = columns_left & columns_right
        # a = {x, y} & b = {y, z} -> columns_map = {left: {x: x}, right: {z: z}}
        self.columns_map = {
            "left": {x:x for x in (columns_left-columns_intersect)},
            "right": {x:x for x in (columns_right-columns_intersect)}
        }

        for col in columns_intersect:
            self.columns_map["left"][col] = f"{self.child_left.table.table_name}.{col}"
            self.columns_map["right"][col] = f"{self.child_right.table.table_name}.{col}"

        columns: list[SQLColumn] = []
        for col in self.child_left.table.table_structure:
            col_name_new = self.columns_map["left"][col.column_name]
            columns.append(SQLColumn(col_name_new, col.column_type, col.primary_key, col.foreign_key))

        for col in self.child_right.table.table_structure:
            col_name_new = self.columns_map["right"][col.column_name]
            columns.append(SQLColumn(col_name_new, col.column_type, col.primary_key, col.foreign_key))

        super().__init__(SQLTable("", name, columns), min(self.child_right.num_tuples, self.child_right.num_tuples))

    def __next__(self) -> dict:
        while True:
            if self.current_tuple is None:
                self.current_tuple = next(self.child_left)

            try:
                rec_right = next(self.child_right)
                joined_record = self._build_joined_record(self.current_tuple, rec_right)
                if self.criteria.eval(joined_record):
                    return joined_record

            except StopIteration:
                self.current_tuple = None
                self.child_right.open()

    def __str__(self) -> str:
        return f"{self.get_description()}({self.child_left}, {self.child_right})"

    def open(self) -> None:
        self.child_left.open()
        self.child_right.open()

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.child_left.close()
        self.child_right.close()

    def get_description(self) -> str:
        return f"⋈_{ str(self.criteria) }"

    def get_structure(self) -> tuple[str, list] | str:
        structure_left = self.child_left.get_structure()
        structure_right = self.child_right.get_structure()
        return super().get_structure(), [structure_left, structure_right]

    def _remap_record(self, side: str, rec: dict):
        return {self.columns_map[side][k]: v for k, v in rec.items()}

    def _build_joined_record(self, left: dict, right: dict) -> dict:
        return self._remap_record("left", left) | self._remap_record("right", right)


class InnerHashJoin(Join):
    def __init__(self, child_left: Operator, child_right: Operator, column_left: Column, column_right: Column):
        self.column_left: Column = column_left
        self.column_right: Column = column_right
        self.ht: dict[Any, dict[Any, Any]] = {}
        self.tuple_right: None | tuple = None
        self.tuples_left: None | list[tuple] = None
        self.index_left: None | int = None
        super().__init__(child_left, child_right, None)

    def open(self) -> None:
        self.child_left.open()
        self.ht = {}
        for rec in self.child_left:
            rec = self._remap_record("left", rec)
            key = self.column_left.get(rec)
            self.ht.setdefault(key, []).append(rec)

        self.child_right.open()

    def __next__(self) -> dict:
        while True:
            if self.tuple_right is None:
                self.tuple_right = self._remap_record("right", next(self.child_right))
                key = self.column_right.get(self.tuple_right)

                if key not in self.ht:
                    continue

                self.tuples_left = self.ht[key]
                self.index_left = 0

            try:
                tuple_left = self.tuples_left[self.index_left]
                self.index_left += 1
                return tuple_left | self.tuple_right

            except IndexError:
                self.tuple_right = None

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"⋈_{{{self.column_left.name} = {self.column_right.name}}}"


class InnerTFIDFJoin(Join):
    # TODO: Implement for comparison (from sklearn.feature_extraction.text import TfidfVectorizer)
    pass


class InnerFuzzyJoin(Join):
    # Fuzzy string matching (Levenshtein, Jaccard, Jaro-Winkler)
    pass


class InnerSoftJoin(Join):
    # TODO: add Attributes to System Prompt
    ZERO_SHOT_SYSTEM_PROMPT = "Given the attributes of the two records, are they the same?. Answer with \"yes\" and \"no\" only!"
    ZERO_SHOT_PROMPTING_TEMPLATE = "Record A is {a}\nRecord B is {b}"

    @staticmethod
    def default_serialization_embedding(x: dict) -> str:
        return '{' + ', '.join([f'{k.split(".")[-1]}: \'{v}\'' for k, v in x.items() if v is not None]) + '}'

    @staticmethod
    def default_serialization_zero_shot_prompting(x: dict) -> str:
        return ', '.join([v for k, v in x.items() if v is not None])

    def __init__(
            self,
            child_left: Operator,
            child_right: Operator,
            method: Literal['threshold', 'zero-shot-prompting', 'both'] = "both",
            embedding_method: Literal["FULL_SERIALIZED", "FIELD_SERIALIZED"] = "FULL_SERIALIZED",
            serialization_embedding: Callable[[dict], str] = default_serialization_embedding,
            vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP,
            threshold: float = None,
            columns_left: None | list[str] = None,
            columns_right: None | list[str] = None,
            em: EmbeddingModel = None,
            sv: SemanticValidationModel = None,
            zs_system_prompt = ZERO_SHOT_SYSTEM_PROMPT,
            zs_template = ZERO_SHOT_PROMPTING_TEMPLATE,
            serialization_zero_shot_prompting: Callable[[dict], str] = default_serialization_zero_shot_prompting):

        self.child_left: Operator = child_left
        self.child_right: Operator = child_left
        self.method: str = method

        if self.method in ("threshold", "both"):
            assert em is not None
            assert threshold is not None
            self.em = em
            self.threshold = threshold
            self.embedding_method: str = embedding_method
            self.vector_store = vector_store_type(em.get_embedding_size())
            self.serialization_embedding: Callable[[dict], str] = serialization_embedding

        if self.method in ("zero-shot-prompting", "both"):
            assert sv is not None
            self.sv = sv
            self.zs_system_prompt = zs_system_prompt
            self.zs_template = zs_template
            self.serialization_zero_shot_prompting = serialization_zero_shot_prompting


        self.embeddings: np.array = None
        self.embeddings_map: dict[Any, np.array] = {}

        self.columns_left: None | list[str] | list[SQLColumn] = None
        self.columns_left: None | list[str] | list[SQLColumn] = None

        self.records_left: Optional[list[dict]] = []
        self.record_right: Optional[dict] = None

        self.distances: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.index_left: Optional[int] = None

        super().__init__(child_left, child_right, None)

        columns_left, columns_right = self._build_join_columns(columns_left, columns_right)
        self.columns_left: list[SQLColumn] = columns_left
        self.columns_right: list[SQLColumn] = columns_right


    def open(self) -> None:
        self.child_left.open()

        logging.debug("Loading records")
        self.records_left = [self._remap_record("left", row) for row in self.child_left]

        if self.method in ("threshold", "both") and len(self.columns_left) > 0:
            self.vector_store.reset()

            logging.debug("Embedding records")
            self.embeddings = self._create_left_embeddings()

            logging.debug("Save to vector store")
            # noinspection PyArgumentList
            self.vector_store.add(self.embeddings)

        self.child_left.close()
        self.child_right.open()

    def __next__(self) -> dict:
        if len(self.records_left) == 0:
            self.close()
            raise StopIteration

        while True:
            if self.record_right is None:
                self.record_right = self._remap_record("right", next(self.child_right))

                if self.method in ("threshold", "both"):
                    if self.embedding_method == "FULL_SERIALIZED":
                        key = str({col.column_name: self.record_right[col.column_name] for col in self.columns_right})
                        embedding_right = self.em(key)
                    else:
                        key_elements = list({str(self.record_right[col.column_name]) for col in self.columns_right})
                        key_elements = list(km for km in key_elements if km not in self.embeddings_map)
                        if len(key_elements) > 0:
                            for km, emb in zip(key_elements, self.em(key_elements)):
                                self.embeddings_map[km] = emb
                        embedding_right = np.average([self.embeddings_map[str(self.record_right[col.column_name])] for col in self.columns_right], axis=0)

                    # noinspection PyArgumentList
                    _, distances, indices = self.vector_store.range_search(x=np.array([embedding_right]), thresh=self.threshold)
                    self.indices = list(indices)
                    self.distances = list(distances)
                else:
                    self.indices = [i for i in range(len(self.columns_left))]

                self.index_left = 0

            try:
                idx = self.indices[self.index_left]
                distance = self.distances[self.index_left]

                self.index_left += 1
                rec = self.records_left[idx] | self.record_right

                logging.debug(f"Joined record {rec} distance {distance}")

                if self.method in ("zero-shot-prompting", "both"):
                    rec_a = {col.column_name: self.records_left[idx][col.column_name] for col in self.columns_left}
                    rec_b = {col.column_name: self.record_right[col.column_name] for col in self.columns_right}
                    prompt = self.zs_template.format(a = self.serialization_zero_shot_prompting(rec_a), b = self.serialization_zero_shot_prompting(rec_b))
                    if not self.sv(prompt, system_prompt=self.zs_system_prompt):
                        logging.debug(f"Prompt: \"{prompt}\" filed in semantic validation")
                        continue

                return rec

            except IndexError:
                self.record_right = None

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"⋈_{{{', '.join(map(str, self.columns_left))} ≈ {', '.join(map(str, self.columns_right))}}}"

    def _build_join_columns(self, columns_left, columns_right):
        # TODO: Soft Search For Join Columns
        join_columns_left: list[SQLColumn] = []
        join_columns_right: list[SQLColumn] = []

        if columns_left is None:
            columns_left = [self.columns_map["left"][col.column_name] for col in self.child_left.table.table_structure]
        else:
            columns_left = [self.columns_map["left"][col] if col in self.columns_map["left"] else col for col in columns_left]

        if columns_right is None:
            columns_right = [self.columns_map["right"][col.column_name] for col in self.child_right.table.table_structure]
        else:
            columns_right = [self.columns_map["right"][col] if col in self.columns_map["right"] else col for col in columns_right]

        columns_found = set({})
        for col in self.table.table_structure:
            if col.column_name in columns_left:
                join_columns_left.append(col)
                columns_found.add(col.column_name)

            if col.column_name in columns_right:
                join_columns_right.append(col)
                columns_found.add(col.column_name)

        missing_columns_left = [col for col in columns_left if col not in columns_found]
        missing_columns_right = [col for col in columns_left if col not in columns_found]

        assert len(missing_columns_left) == 0, \
            f"Columns {missing_columns_left} not found for right relation {self.table.table_structure}"
        assert len(missing_columns_right) == 0, \
            f"Columns {missing_columns_right} not found for left relation {self.table.table_structure}"

        return join_columns_left, join_columns_right

    def _create_left_embeddings(self) -> np.array:
        if self.embedding_method == "FULL_SERIALIZED":
            keys = [self.serialization_embedding({col.column_name: rec[col.column_name] for col in self.columns_left}) for rec in self.records_left]
            return self.em(keys)
        else:
            key_elements_set = list({str(rec[col.column_name]) for col in self.columns_left for rec in self.records_left})
            self.embeddings_map = {key: embedding for key, embedding in zip(key_elements_set, self.em(key_elements_set))}
            key_elements_embeddings = np.array([
                [self.embeddings_map[str(rec[col.column_name])] for col in self.columns_left]
                for rec in self.records_left
            ])
            return np.average(key_elements_embeddings, axis=1)
