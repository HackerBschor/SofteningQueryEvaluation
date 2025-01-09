from typing import Callable

import numpy as np

from operators import Operator
from utils import get_idx_closes_vector, Measure, EuclidianDistance
from utils.DB import DBConnector
from utils.Model import EmbeddingModel


class Scan(Operator):
    def __init__(self, name: str, db_connector: DBConnector, embedding_model: EmbeddingModel,
                 num_tuples: int = 10,  batch_size: int = 100,
                 threshold: float = 100.0,
                 distance: Measure = EuclidianDistance()) -> None:

        self.name: str = name
        self.threshold: float = threshold
        self.batch_size: int = batch_size
        self.distance: Measure = distance
        self.db_connector: DBConnector = db_connector
        self.embedding_model: EmbeddingModel = embedding_model
        # self.generation_model: GenerationModel = generation_model
        self.schema_name, self.table_name = self._get_table()

        assert self.schema_name is not None, f"No table found for '{name}'"

        self.cursor = self.db_connector.get_cursor()
        self.cursor.execute(f"SELECT * FROM {self.schema_name}.{self.table_name}")
        super().__init__(name, [desc[0] for desc in self.cursor.description], num_tuples)


    def __str__(self) -> str:
        return self.name

    def open(self):
        if self.cursor is not None:
            self.cursor.close()

        self.cursor = self.db_connector.get_cursor()
        self.cursor.execute(f"SELECT * FROM {self.schema_name}.{self.table_name}")


    def __next__(self) -> dict:
        try:
            return next(self.cursor)
        except StopIteration:
            self.close()
            raise StopIteration


    def next_vectorized(self) -> list[dict]:
        idx: int = 0
        return_data: list[dict] = []

        try:
            while idx < self.num_tuples:
                return_data.append(next(self.cursor))
                idx += 1
        except StopIteration:
            pass

        return None if len(return_data) == 0 else return_data

    def close(self):
        self.cursor.close()

    def _get_table(self) -> (str, str):
        """
        Searches through all schemas and tables to find the closest match for the init name
        :return: schema name and table name
        """
        with self.db_connector.get_cursor() as cursor:
            cursor.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name;
            """)
            res = cursor.fetchall()

        embeddings = self.embedding_model.embedd_batch([self.name] + list(map(lambda x: x["table_name"], res)))

        idx: int | None = get_idx_closes_vector(self.distance, self.threshold, embeddings[1:], embeddings[0])

        return res[idx]["table_schema"], res[idx]["table_name"] if idx is not None else None
