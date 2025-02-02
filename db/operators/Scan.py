import logging
from typing import Type

import re

import faiss
import numpy as np

from .Operator import Operator
from ..structure import SQLTable
from ..db import DBConnector
from models.embedding.Model import EmbeddingModel
from models.semantic_validation.Model import SemanticValidationModel



class Scan(Operator):
    TABLE_SCHEMA_PATTERN = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'

    def __init__(
            self, name: str, db: DBConnector, em: EmbeddingModel, sv: SemanticValidationModel,
            num_tuples: int = 10, batch_size: int = 100, threshold: float = .8,
            vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP, limit: int = None,
            sql_annex: str | None = None, use_semantic_table_search: bool = True,
            use_semantic_validation: bool = False,
    ) -> None:
        self.name: str = name
        self.threshold: float = threshold
        self.batch_size: int = batch_size
        self.limit = limit
        self.db: DBConnector = db
        self.em: EmbeddingModel = em
        self.sv: SemanticValidationModel = sv
        self.use_semantic_table_search = use_semantic_table_search
        self.vector_store = vector_store_type(self.em.get_embedding_size())
        self.is_distance = self.vector_store.metric_type == faiss.METRIC_L2

        self.use_semantic_validation = use_semantic_validation
        table, confidence = self._get_table()

        assert table is not None, "No table found"
        logging.debug(f"Selected Table (confidence {confidence:.02}): {table}")

        self.cursor = None
        self.query = f"SELECT * FROM {table.table_schema}.{table.table_name}"

        if sql_annex is not None:
            self.query += f" {sql_annex}"

        if limit is not None:
            self.query += f" LIMIT {limit}"

        super().__init__(table, num_tuples)


    def __str__(self) -> str:
        return f'"{self.name}"≈>"{self.table.table_schema}.{self.table.table_name}"'

    def open(self):
        if self.cursor is not None:
            self.cursor.close()

        self.cursor = self.db.get_cursor()
        self.cursor.execute(self.query)


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

    def _get_table(self) -> (SQLTable | None, float):
        """
        Searches through all schemas and tables to find the closest match for the init name
        :return: schema name and table name
        """

        if not self.use_semantic_table_search:
            result = re.match(self.TABLE_SCHEMA_PATTERN, self.name)
            assert result, "Kein Schema.Table angegeben"
            table_schema, table_name = result.group(1), result.group(2)
            return self.db.tables[f"{table_schema}.{table_name}"], 1.0

        name_input = f"SQL Table for '{self.name}' (structure: <schame>.<name>: [<column>(<type>[, PRIMARY_KEY, VALUE_SAMPLES(<values>)])])"
        table_names = [str(table) for table in self.db.tables]
        embeddings = self.em([name_input] + table_names)

        # noinspection PyArgumentList
        self.vector_store.add(embeddings[1:])

        # noinspection PyArgumentList
        distances, idxs = self.vector_store.search(np.array([embeddings[0]]), len(embeddings) - 1)

        # TODO: What to do for more than one Table
        for i in range(len(embeddings) - 1):
            idx, distance = idxs[0][i], distances[0][i]

            if distance < self.threshold if self.is_distance else distance > self.threshold:
                return None, None

            sql_table = self.db.tables[table_names[idx]]

            if not self.use_semantic_validation:
                return sql_table, distance

            prompt = f"Does this SQL Table '{sql_table}' describe entities for '{self.name}'?"

            logging.debug(prompt)

            if self.sv(prompt):
                return sql_table, distance

        raise Exception("Table not found")