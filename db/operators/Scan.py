import logging
from typing import Type

import faiss
import numpy as np

from .Operator import Operator
from ..structure import SQLTable
from utils.DB import DBConnector
from models.embedding.Model import EmbeddingModel
from models.semantic_validation.Model import SemanticValidationModel

import re

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

    TABLE_SCHEMA_PATTERN = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'

    def __init__(
            self, name: str, db: DBConnector, em: EmbeddingModel, sv: SemanticValidationModel,
            num_tuples: int = 10, batch_size: int = 100, threshold: float = .8,
            vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP, limit: int = None,
            sql_annex: str | None = None, use_semantic_table_search: bool = True,
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
        table, confidence = self._get_table()

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
        sql_tables = []
        with self.db.get_cursor() as cursor:
            if self.use_semantic_table_search:
                cursor.execute(self.SQL_FETCH_TABLES)
                for row in cursor.fetchall():
                    sql_tables.append(SQLTable(row["table_schema"], row["table_name"], row["table_structure"]))
            else:
                result = re.match(self.TABLE_SCHEMA_PATTERN, self.name)
                assert result, "Kein Schema.Table angegeben"
                table_schema, table_name = result.group(1), result.group(2)
                cursor.execute(
                    f"SELECT * FROM ({ self.SQL_FETCH_TABLES }) x WHERE table_schema = %s AND table_name = %s",
                    (table_schema, table_name))
                row = cursor.fetchone()
                return SQLTable(row["table_schema"], row["table_name"], row["table_structure"]), 1.0

        name_input = f"SQL Table for '{self.name}' (structure: <schame>.<name>: [<column>(<type>[, PRIMARY_KEY])])"
        table_inputs = list(map(str, sql_tables))

        embeddings = self.em([name_input] + table_inputs)

        # noinspection PyArgumentList
        self.vector_store.add(embeddings[1:])

        # noinspection PyArgumentList
        distances, idxs = self.vector_store.search(np.array([embeddings[0]]), len(embeddings) - 1)

        # TODO: What to do for more than one Table
        for i in range(len(embeddings) - 1):
            idx, distance = idxs[0][i], distances[0][i]
            table = sql_tables[idx]
            prompt = f"Does this SQL Table '{table}' describe entities for '{self.name}'?"
            logging.debug(prompt)

            if self.sv(prompt):
                return table, distance

        raise Exception("Table not found")