from typing import List, Type

import faiss
import numpy as np

from operators import Operator, SQLTable

from utils.Model import EmbeddingModel

class Project(Operator):
    def __init__(self, child_operator: Operator, columns: list[str], em: EmbeddingModel,
                 vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP, threshold: float = 0) -> None:

        self.child_operator: Operator = child_operator
        self.columns = columns
        vector_store = vector_store_type( em.get_embedding_shape())

        columns_available = [col.column_name for col in self.child_operator.table.table_structure]
        embeddings = em.embedd_batch(columns + columns_available)

        # noinspection PyArgumentList
        vector_store.add( embeddings[len(columns):])

        selected_columns = []
        for i, (col, emb) in enumerate(zip(columns, embeddings[:len(columns)])):
            # noinspection PyArgumentList
            _, distances, indices = vector_store.range_search(x=np.array([emb]), thresh=threshold)
            closest_column_index = indices[np.argmax(distances)]
            selected_columns.append(self.child_operator.table.table_structure[closest_column_index])

        t = SQLTable(child_operator.table.table_schema, child_operator.table.table_name, selected_columns)
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
        col_desc = [f"{name}≈{col.column_name}" for name, col in zip(self.columns, self.table.table_structure)]
        return f"π_{{{', '.join(col_desc)}}}"

    def get_structure(self) -> tuple[str, List] | str:
        return super().get_structure(), [self.child_operator.get_structure()]

    def _remap_dict(self, r):
        return {col.column_name: r[col.column_name] for col in self.table.table_structure}