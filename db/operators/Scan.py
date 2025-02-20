import logging
import re

import faiss
import numpy as np

from db.structure import SQLTable
from db.db import DBConnector
from db.operators.Operator import Operator
from models.embedding.Model import EmbeddingModel
from models.semantic_validation.Model import SemanticValidationModel

from typing import Type


class Scan(Operator):
    """
    This class performs a semantic search for a table in a database.
    In the interation, it returns the records from the table.

    Attributes:
        name (str): The name that should be used to identify the table
        db (db.db.DBConnector): A database connector
        em (EmbeddingModel): The model to embedd a serialized database table
        sv (SemanticValidationModel): The model which performs a sematnic validation if the name matches with the table
        vector_index_class (faiss.IndexFlat): Class to instantiate a vector index
            (Default: faiss.IndexFlatIP -> CosineSimilarity)
        threshold (float): Threshold for the similarity between name embedding and table embedding
        sql_annex (Optional[str]): If set, appends SQL to the selection query
            "SELECT <schem>.<table> <sql_annex>
        limit (Optional[int]): If set, appends a limit to the table selection query
            "SELECT <schem>.<table> LIMIT <limit>"
        use_semantic_table_search (bool): Should the table be searched semantically?
        use_semantic_validation (bool): After semantic search, should am LLM validate that the table matches the name
            -> PerformsLLM.prompt("Does this SQL Table '<schem>.<table> Columns: ...' describe entities for '<name>'"
    """

    # Pattern to validate a name in the format: "<schema>.<table>"
    TABLE_SCHEMA_PATTERN = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'

    def __init__(
            self, name: str, db: DBConnector,
            use_semantic_table_search: bool = True, em: EmbeddingModel = None, threshold: float = .8,
            use_semantic_validation: bool = True, sv: SemanticValidationModel = None,
            vector_index_class: Type[faiss.IndexFlat] = faiss.IndexFlatIP,
            limit: str | int = None,
            sql_annex: str | None = None) -> None:

        self.name: str = name
        self.db: DBConnector = db

        self.use_semantic_table_search = use_semantic_table_search
        self.use_semantic_validation = use_semantic_validation

        if self.use_semantic_table_search:
            self.em: EmbeddingModel = em
            self.threshold: float = threshold
            self.vector_index = vector_index_class(self.em.get_embedding_size())  # Instantiate vector index
            # IndexFlatIP ->CosineSimilarity => Similarity, L2 -> Euclidian Distance => Distance
            self.is_distance = self.vector_index.metric_type == faiss.METRIC_L2

        if self.use_semantic_validation:
            self.sv: SemanticValidationModel = sv

        table, confidence = self._get_table() # Semantic Search for table

        assert table is not None, "No table found"

        logging.debug(f"Selected Table (confidence {confidence:.02}): {table}")

        # Build "SELECT <schem>.<table> <sql_annex> LIMIT <limit>" Query
        self.cursor = None
        self.query = f"SELECT * FROM {table.table_schema}.{table.table_name}"
        self.query += f" {sql_annex}" if sql_annex is not None else ""
        self.query += f" LIMIT {limit}" if limit is not None else ""

        super().__init__(table, 10)

    def __str__(self) -> str:
        return f'"{self.name}"≈>"{self.table.table_schema}.{self.table.table_name}"'

    def open(self) -> Operator:
        if self.cursor is not None:
            self.cursor.close()

        # Fetch data from PostgreSQL
        self.cursor = self.db.get_cursor()
        self.cursor.execute(self.query)

        return self

    def __next__(self) -> dict:
        # If available: return next record from the data, otherwise: close connection

        try:
            return next(self.cursor)
        except StopIteration:
            self.close()
            raise StopIteration

    def next_vectorized(self) -> list[dict]:
        # If available: return next *n* record from the data, otherwise: close connection

        idx: int = 0
        return_data: list[dict] = []

        try:
            while idx < self.num_tuples:
                return_data.append(next(self.cursor))
                idx += 1
        except StopIteration:
            pass

        return None if len(return_data) == 0 else return_data

    def close(self) -> None:
        self.cursor.close()

    def _get_table(self) -> (SQLTable | None, float):
        """
        Semantic Search for schemas.tables to find the closest match for the stated name
        :return: schema name and table name
        """

        if not self.use_semantic_table_search:
            # Table schema.table is known -> No semantic search
            result = re.match(self.TABLE_SCHEMA_PATTERN, self.name)
            assert result, "Kein Schema.Table angegeben"
            table_schema, table_name = result.group(1), result.group(2)
            return self.db.tables[f"{table_schema}.{table_name}"], 1.0

        # Serialize name for embedding
        name_input = f"SQL Table for '{self.name}' "
        name_input += "(structure: <schame>.<name>: [<column>(<type>[, PRIMARY_KEY, VALUE_SAMPLES(<values>)])])"

        # Embedd serialized name and tables and isnert embeddings in the vector index
        table_names = [str(table) for table in self.db.tables]
        embeddings = self.em([name_input] + table_names)

        # noinspection PyArgumentList
        self.vector_index.add(embeddings[1:])

        # noinspection PyArgumentList
        _, distances, idxs = self.vector_index.range_search(np.array([embeddings[0]]), thresh=self.threshold)
        assert len(distances) > 0, f"No table found for name '{self.name}', available tables: {table_names}"

        # np.argsort sorts ascending -> good for distance, if its similarity score: sort descending
        indices = np.argsort(distances)
        if not self.is_distance:
            indices = indices[::-1]

        # Retrieve tables with the closest embedding to name
        for i in indices:
            distance, idx = distances[i], idxs[i]

            sql_table = self.db.tables[table_names[idx]]

            if not self.use_semantic_validation:
                return sql_table, distance

            # Semantic validate using LLM
            #f"Does this SQL Table '{sql_table.table_schema}.{sql_table.table_name}' describe entities for '{self.name}'?"
            prompt = (f"Does the table name '{sql_table.table_schema}.{sql_table.table_name}'"
                f"match the intent of the search query '{self.name}'"
                "\nConsider semantic similarity, synonyms, abbreviations, and language variations."
            )

            if self.sv(prompt):
                return sql_table, distance

        raise Exception(f"No table found for name '{self.name}', available tables: {table_names}")
