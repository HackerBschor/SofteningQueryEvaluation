import faiss
import numpy as np
import logging

from typing import Any, Type, Optional

from .Operator import Operator
from ..structure import SQLColumn, SQLTable, Column
from ..criteria import Criteria
from models.embedding.Model import EmbeddingModel


class Join(Operator):
    def __init__(self, child_left: Operator, child_right: Operator, criteria: Criteria | None):
        self.child_left: Operator = child_left
        self.child_right: Operator = child_right
        self.criteria: Criteria = criteria
        self.current_tuple = None

        name = f"{self.child_left.table.table_name}+{self.child_left.table.table_name}"

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


class InnerSoftJoin(Join):
    def __init__(
            self,
            child_left: Operator,
            child_right: Operator,
            column_left: Column | None | list[str],
            column_right: Column | None | list[str],
            embedding_mode: EmbeddingModel,
            vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP,
            threshold: float = 0,
            debug = False):

        # TODO: Soft Search For Join Columns

        self.column_left: Column | None | list[str] = column_left
        self.column_right: Column | None | list[str] = column_right
        self.embedding_mode: EmbeddingModel = embedding_mode
        self.vector_store_type = vector_store_type
        self.threshold: float = threshold
        self.debug: bool = debug

        self.vector_store: Optional[Type[faiss.IndexFlat]] = None
        self.records_left: Optional[list[dict]] = []
        self.record_right: Optional[dict] = None

        self.distances: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.index_left: Optional[int] = None

        super().__init__(child_left, child_right, None)

    def open(self) -> None:
        self.child_left.open()
        self.vector_store = self.vector_store_type(self.embedding_mode.get_embedding_size())

        logging.debug("Loading & embedd records")
        embeddings = []
        for row in self.child_left:
            row = self._remap_record("left", row)
            self.records_left.append(row)
            embeddings.append(self._create_embedding(row, "left"))

        if embeddings:
            logging.debug("Save to vector store")
            # noinspection PyArgumentList
            self.vector_store.add(np.array(embeddings))

        self.child_left.close()
        self.child_right.open()

    def __next__(self) -> dict:
        if not self.records_left:
            self.close()
            raise StopIteration

        while True:
            if self.record_right is None:
                self.record_right = self._remap_record("right", next(self.child_right))
                embedding_right = self._create_embedding(self.record_right, "right")

                # noinspection PyArgumentList
                _, distances, indices = self.vector_store.range_search(x=np.array([embedding_right]), thresh=self.threshold)
                self.indices = list(indices)
                self.distances = list(distances)
                self.index_left = 0

            try:
                idx = self.indices[self.index_left]
                distance = self.distances[self.index_left]

                # if idx < 0 or distance > self.threshold:
                #     raise IndexError

                self.index_left += 1

                # TODO: Semantic Validation

                return self.records_left[idx] | self.record_right | ({} if not self.debug else {"distance": distance })

            except IndexError:
                self.record_right = None

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"⋈_{{{self.column_left.name} ≈ {self.column_right.name}}}"

    def _create_embedding(self, rec, side):
        column = self.column_left if side == "left" else self.column_right

        if column is None:
            key = str(rec)
        elif isinstance(column, Column):
            key = column.get(rec)
        else:
            key = ", ".join((str(rec[col]) for col in column))

        return self.embedding_mode.embedd(key)[0]
