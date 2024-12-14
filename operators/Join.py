import faiss
import numpy as np

from typing import List, Any

from operators import Operator, Column
from utils.Model import EmbeddingModel, GenerationModel


class Join(Operator):
    def __init__(self, child_left: Operator, child_right: Operator):
        self.child_left: Operator = child_left
        self.child_right: Operator = child_right
        name = f"{self.child_left.name}+{self.child_right.name}"
        columns = [f"{self.child_left.name}.{col}" for col in self.child_left.columns]
        columns += [f"{self.child_right.name}.{col}" for col in self.child_right.columns]
        super().__init__(name, columns)

    def __next__(self) -> dict:
        raise NotImplemented()

    def get_structure(self) -> tuple[str, List] | str:
        structure_left = self.child_left.get_structure()
        structure_right = self.child_right.get_structure()
        return super().get_structure(), [structure_left, structure_right]

class InnerHashJoin(Join):
    def __init__(self, child_left: Operator, child_right: Operator, column_left: Column, column_right: Column):
        self.column_left: Column = column_left
        self.column_right: Column = column_right
        self.ht: None | dict[Any, dict[Any, Any]] = None
        self.tuple_right: None | tuple = None
        self.tuples_left: None | List[tuple] = None
        self.index_left: None | int = None
        super().__init__(child_left, child_right)

    def __str__(self):
        return f"⋈({self.child_left.name}.{self.column_left.name}={self.child_right.name}.{self.column_right.name})"

    def __next__(self) -> dict:
        if self.ht is None:
            self.build_ht()

        while True:
            if self.tuple_right is None:
                tuple_right = next(self.child_right)
                key = self.column_right.get(tuple_right)

                if key not in self.ht:
                    continue

                self.tuples_left = self.ht[key]
                self.tuple_right = {f"{self.child_right.name}.{key}": value for key, value in tuple_right.items()}
                self.index_left = 0

            try:
                tuple_left = self.tuples_left[self.index_left]
                self.index_left += 1
                return tuple_left | self.tuple_right

            except IndexError:
                self.tuple_right = None


    def build_ht(self):
        self.ht = {}
        for t in self.child_left:
            key = self.column_left.get(t)
            t_new = {f"{self.child_left.name}.{key}": value for key, value in t.items()}
            if key in self.ht:
                self.ht[key].append(t_new)
            else:
                self.ht[key] = [t_new]


class SoftInnerJoin(Join):
    # PROMPT_TEMPLATE = "Do the descriptions '{}' & '{}' describe the same entity?"

    def __init__(
            self,
            child_left: Operator,
            child_right: Operator,
            column_left: Column,
            column_right: Column,
            embedding_mode: EmbeddingModel,
            generation_mode: GenerationModel,
            threshold: float = 1):

        self.column_left: Column = column_left
        self.column_right: Column = column_right
        self.embedding_mode = embedding_mode
        self.generation_mode = generation_mode
        self.threshold: float = threshold

        self.vector_store: None | faiss.IndexFlatL2 = None
        self.tuples_left: None | List[dict] = None

        self.tuple_right: None | dict = None

        self.distances: None | np.ndarray = None
        self.indices: None | np.ndarray = None
        self.index_left: None | int = None

        super().__init__(child_left, child_right)


    def __str__(self):
        return f"⋈({self.child_left.name}.{self.column_left.name}≈{self.child_right.name}.{self.column_right.name})"


    def __next__(self) -> dict:
        if self.vector_store is None:
            self.build_vector_store()

        while True:
            if self.tuple_right is None:
                tuple_right = next(self.child_right)
                key = self.column_right.get(tuple_right)
                key_embedding = self.embedding_mode.embedd(key)
                # noinspection PyArgumentList
                self.distances, self.indices = self.vector_store.search(key_embedding, len(self.tuples_left))

                self.tuple_right = {f"{self.child_right.name}.{key}": value for key, value in tuple_right.items()}
                self.index_left = 0

            try:
                idx = self.indices[0][self.index_left]
                distance = self.distances[0][self.index_left]

                if idx < 0 or distance > self.threshold:
                    raise IndexError

                self.index_left += 1
                new_tuple = self.tuples_left[idx] | self.tuple_right
                # key_left = self.tuples_left[idx][f"{self.child_left.name}.{self.column_left.name}"]
                # key_right = self.tuples_left[idx][f"{self.child_left.name}.{self.column_left.name}"]
                # prompt = self.PROMPT_TEMPLATE.format(key_left, key_right)
                # result = self.generation_mode.generate(prompt, max_new_tokens=3)
                return new_tuple

            except IndexError:
                self.tuple_right = None

    def build_vector_store(self):
        self.vector_store = faiss.IndexFlatL2(self.embedding_mode.embedd("Test").shape[1])
        tuples_left = list(self.child_left)
        keys_left = list(map(lambda x: self.column_left.get(x), tuples_left))
        self.tuples_left = [{f"{self.child_left.name}.{key}": value for key, value in row.items()} for row in tuples_left]
        embeddings_left = self.embedding_mode.embedd(keys_left)
        # noinspection PyArgumentList
        self.vector_store.add(embeddings_left)
