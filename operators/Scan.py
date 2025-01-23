import logging
from typing import Type

import faiss
import numpy as np

from operators import Operator, SQLTable
from utils.DB import DBConnector
from utils.Model import EmbeddingModel

class Scan(Operator):
    SQL_FETCH_TABLES = """
            WITH primary_keys AS (
                SELECT tc.table_name, tc.table_schema, kcu.column_name, ':PRIMARY_KEY' AS prim
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu USING (constraint_name, table_schema)
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ), foregin_keys AS (
                SELECT tc.table_schema, tc.table_name, kcu.column_name,
                       CONCAT(':FOREIGN_KEY(', ccu.table_schema, '.', ccu.table_name, '.', ccu.column_name, ')') AS foreign_table
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu USING (constraint_name, table_schema)
                JOIN information_schema.constraint_column_usage AS ccu USING (constraint_name)
                WHERE tc.constraint_type = 'FOREIGN KEY'
            )
            SELECT
                table_schema, table_name, STRING_AGG(CONCAT(
                    c.column_name, ':', c.data_type, COALESCE(p.prim, ''), COALESCE(f.foreign_table, '')
                ), ', ') AS table_structure
            FROM information_schema.tables t
            JOIN information_schema.columns c USING (table_schema, table_name)
            LEFT JOIN primary_keys p USING (table_schema, table_name, column_name)
            LEFT JOIN foregin_keys f USING (table_schema, table_name, column_name)
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            GROUP BY table_schema, table_name
            ORDER BY table_schema, table_name
    """

    def __init__(
            self, name: str, db_connector: DBConnector, embedding_model: EmbeddingModel,
            num_tuples: int = 10, batch_size: int = 100, threshold: float = .8,
            vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP, limit: int = None,
            sql_annex: str | None = None,
    ) -> None:
        self.name: str = name
        self.threshold: float = threshold
        self.batch_size: int = batch_size
        self.limit = limit
        self.db_connector: DBConnector = db_connector
        self.embedding_model: EmbeddingModel = embedding_model
        self.vector_store = vector_store_type(self.embedding_model.get_embedding_shape())
        table, confidence = self._get_table()
        self.cursor = None

        self.query = f"SELECT * FROM {table.table_schema}.{table.table_name}"

        if sql_annex is not None:
            self.query += f" {sql_annex}"

        if limit is not None:
            self.query += f" LIMIT {limit}"


        logging.debug(f"Selected Table (confidence {confidence:.02}): {table}")

        assert table is not None, f"No table found for '{name}'"

        super().__init__(table, num_tuples)


    def __str__(self) -> str:
        return self.name

    def open(self):
        if self.cursor is not None:
            self.cursor.close()

        self.cursor = self.db_connector.get_cursor()
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
        sql_tables = []
        with self.db_connector.get_cursor() as cursor:
            cursor.execute(self.SQL_FETCH_TABLES)
            for row in cursor.fetchall():
                sql_tables.append(SQLTable(row["table_schema"], row["table_name"], row["table_structure"]))

        name_input = f"SQL Table for '{self.name}' (structure: <schame>.<name>: [<column>(<type>[, PRIMARY_KEY])])"
        table_inputs = list(map(str, sql_tables))

        embeddings = self.embedding_model.embedd_batch([name_input] + table_inputs)

        # noinspection PyArgumentList
        self.vector_store.add(embeddings[1:])

        # noinspection PyArgumentList
        distances, idxs = self.vector_store.search(np.array([embeddings[0]]), 1)

        distance, idx = distances[0][0], idxs[0][0]

        return sql_tables[idx], distance
