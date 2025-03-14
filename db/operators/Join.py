import faiss
import numpy as np
import logging

from typing import Any, Type, Optional, Literal, Callable

from db.operators.Operator import Operator
from db.structure import SQLColumn, SQLTable, Column
from db.criteria import Criteria

from models.embedding.Model import EmbeddingModel
from models.semantic_validation.Model import SemanticValidationModel


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class Join(Operator):
    """
    Implementation of NestedLoopJoin that evaluate a criteria.
    Construct criterion using the criteria in db.criteria.py

    Renames columns with same name from both relations.
    E.g. L.columns = {x, y}, R.columns = {y, z} -> returns tuples of {x, L.y, R.y, z}

    Attributes:
        child_left (Operator): Left Relation
        child_right (Operator): Right Relation
        criteria (Criteria): Join Criteria
    """

    def __init__(self, child_left: Operator, child_right: Operator, criteria: Criteria | None):
        self.child_left: Operator = child_left
        self.child_right: Operator = child_right
        self.criteria: Criteria = criteria
        self.current_tuple = None

        name = f"{self.child_left.table.table_name}+{self.child_right.table.table_name}"

        # Renames columns of the same name:
        # L = {x, y} & R = {y, z} -> columns_map = {left: {x: x, y: L.y}, right: {z: z, y: R.y}}
        columns_left = {x.column_name for x in self.child_left.table.table_structure}
        columns_right = {x.column_name for x in self.child_right.table.table_structure}
        # Intercepting column Names: {y}
        columns_intersect = columns_left & columns_right

        # builds columns map for unambiguous columns {left: {x: x}, right: {z: z}
        self.columns_map = {
            "left": {x:x for x in (columns_left-columns_intersect)},
            "right": {x:x for x in (columns_right-columns_intersect)}
        }

        # Add renames to column map: left.put({y->L.y}), right.put({y->R.y})
        for col in columns_intersect:
            self.columns_map["left"][col] = f"{self.child_left.table.table_name}.{col}"
            self.columns_map["right"][col] = f"{self.child_right.table.table_name}.{col}"

        # Builds new structure for renamed column map
        columns: list[SQLColumn] = []
        for col in self.child_left.table.table_structure:
            col_name_new = self.columns_map["left"][col.column_name]
            columns.append(SQLColumn(col_name_new, col.column_type, col.primary_key, col.foreign_key))

        for col in self.child_right.table.table_structure:
            col_name_new = self.columns_map["right"][col.column_name]
            columns.append(SQLColumn(col_name_new, col.column_type, col.primary_key, col.foreign_key))

        super().__init__(SQLTable("", name, columns), min(self.child_right.num_tuples, self.child_right.num_tuples))

    def __next__(self) -> dict:
        """
        Iterate over left relation and fix the record.
        Iterate over right relation until the criteria is fulfilled
            -> rename (according to columns_map) records -> return merged records
        If the right relation is exhausted, fix next record from left relation and repeat
        If left relation is exhausted -> close()
        """
        while True:
            # Fix next left record
            if self.current_tuple is None:
                self.current_tuple = next(self.child_left)

            try:
                # Gen next record form right relation and check criteria
                rec_right = next(self.child_right)
                joined_record = self._build_joined_record(self.current_tuple, rec_right) # join record & rename columns
                if self.criteria.eval(joined_record):
                    return joined_record

            except StopIteration:
                # Right relation is exhausted -> fix next left record from relation
                self.current_tuple = None
                self.child_right.open()

    def __str__(self) -> str:
        return f"{self.get_description()}({self.child_left}, {self.child_right})"

    def open(self) -> Operator:
        self.child_left.open()
        self.child_right.open()
        return self

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
        """
        Remap record to remove ambiguous columns
        columns_map = {left: {x: x, y: L.y}, right: {z: z, y: R.y}}
        E.g. left record {"x": 1, "y": 2} -> {"x": 1, "L.y": 2}
        """
        return {self.columns_map[side][k]: v for k, v in rec.items()}

    def _build_joined_record(self, left: dict, right: dict) -> dict:
        """
        Remap records to remove ambiguous columns & Merge
        E.g. left={"x": 1, "y": 2}, right={"y": 2, "z": 3} -> {"x": 1, "L.y": 2, "R.y": 2, "z": 3}
        """
        return self._remap_record("left", left) | self._remap_record("right", right)


class InnerHashJoin(Join):
    """
    Implementation of HashJoin on equality of columns.

    Renames columns with same name from both relations.
    E.g. L.columns = {x, y}, R.columns = {y, z} -> returns tuples of {x, L.y, R.y, z}

    Attributes:
        child_left (Operator): Left Relation
        child_right (Operator): Right Relation
        column_left (Column): Left Join Column
        column_right (Column): Right Join Column
    """

    def __init__(self, child_left: Operator, child_right: Operator, column_left: Column, column_right: Column):
        self.column_left: Column = column_left
        self.column_right: Column = column_right
        self.ht: dict[Any, list[dict]] = {}
        self.record_right: None | tuple = None
        self.records_left: None | list[tuple] = None
        self.index_left: None | int = None
        super().__init__(child_left, child_right, None)

    def open(self) -> Operator:
        """ Iterate over left relation and insert tuples in HashTable {JoinValue -> list[records]}"""
        self.child_left.open()
        self.ht = {}
        for rec in self.child_left:
            rec = self._remap_record("left", rec)
            key = self.column_left.get(rec) # HashTable Key
            self.ht.setdefault(key, []).append(rec) # Append record to list for ht[key]

        self.child_right.open()
        return self

    def __next__(self) -> dict:
        """
        Iterate over right relation and fix the record. Load all matching records (left relation) from hash table

        Retrieve next record from the matching records list
            -> rename (according to columns_map) records -> return merged records

        If the matching records list is exhausted, fix next record from right relation and repeat
        If right relation is exhausted -> close()
        """

        while True:
            if self.record_right is None:
                # Fix next right record
                self.record_right = self._remap_record("right", next(self.child_right))
                key = self.column_right.get(self.record_right) # HashTable key

                if key not in self.ht: # No matches found
                    continue

                # Retrieve matching records from left relation
                self.records_left = self.ht[key]
                self.index_left = 0

            try:
                # Retrieve matching record from matching records list
                record_left = self.records_left[self.index_left]
                self.index_left += 1
                return record_left | self.record_right # Return merged records

            except IndexError:
                # matching records list is exhausted -> repeat with next right record
                self.record_right = None

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"⋈_{{{self.column_left.name} = {self.column_right.name}}}"


class InnerTFIDFJoin(Join):
    # TODO: Implement for comparison (from sklearn.feature_extraction.text import TfidfVectorizer)
    pass


class InnerFuzzyJoin(Join):
    # TODO: Implement for comparison
    # Fuzzy string matching (Levenshtein, Jaccard, Jaro-Winkler)
    pass


class InnerSoftJoin(Join):
    """
    Implements the novel SoftJoin Operator, that joins records on semantic equality.

    General Functionality:
    Iterate over left relation and insert the embeddings of serialized records into a vector index.
    Then, iterate over right relation, serialize and embedd record. Then perform a range query on the vector index.
    For all records within a threshold, perform entity matching using an LLM.
    If the entity matching passes, return merged tuple.

    Renames columns with same name from both relations.
    E.g. L.columns = {x, y}, R.columns = {y, z} -> returns tuples of {x, L.y, R.y, z}

    Attributes:
        child_left (Operator): Left Relation
        child_right (Operator): Right Relation
        method (threshold, zero-shot-prompting, both): Rely only on threshold/ Zero shot prompting/ combination
        embedding_comparison (RECORD_WISE, COLUMN_WISE): compare the embeddings for serialized records
            or for every column separately
        embedding_method (FULL_SERIALIZED, FIELD_SERIALIZED): How the record should be serialized
        serialization_embedding: Callable[[dict], str]: Function for serialization (embeddings)
        vector_store_type (faiss.IndexFlat): vector index (E.g. IndexFlatIP -> Cosine Similarity)
        threshold (float): threshold for range query
        columns_left (None | list[str]): Left Join Columns (None -> Serialize entire Record)
        columns_right (None | list[str]): Right Join Columns (None -> Serialize entire Record)
        em (EmbeddingModel): The model to embedd a serialized record
        sv (SemanticValidationModel): The model which performs a semantic validation if both records match
        zs_system_prompt (str): System Prompt for semantic validation,
        zs_template (str): Prompt for semantic validation,
        serialization_zero_shot_prompting: Function for serialization (zero-shot-prompting)
    """

    ZERO_SHOT_PROMPTING_TEMPLATE = 'Does "{a}" describe the same reals world entity as "{b}"'

    @staticmethod
    def default_serialization_embedding(x: dict) -> str:
        return '{' + ', '.join([f'{k.split(".")[-1]}: \'{v}\'' for k, v in x.items() if v is not None]) + '}'

    @staticmethod
    def default_serialization_zero_shot_prompting(x: dict) -> str:
        return ', '.join([str(v) for k, v in x.items() if v is not None])

    def __init__(
        self,
        child_left: Operator,
        child_right: Operator,
        method: Literal['threshold', 'zero-shot-prompting', 'both'] = "both",
        embedding_comparison: Literal["RECORD_WISE", "COLUMN_WISE"] = "RECORD_WISE",
        embedding_method: Literal["FULL_SERIALIZED", "FIELD_SERIALIZED"] = "FULL_SERIALIZED",
        serialization_embedding: Callable[[dict], str] = default_serialization_embedding,
        vector_store_type: Type[faiss.IndexFlat] = faiss.IndexFlatIP,
        threshold: float = None,
        columns_left: None | list[str] = None,
        columns_right: None | list[str] = None,
        em: EmbeddingModel = None,
        sv: SemanticValidationModel = None,
        zs_system_prompt = None,
        zs_template = None,
        serialization_zero_shot_prompting: Callable[[dict], str] | Callable[[dict, dict], str] = None
    ):

        assert method in ['threshold', 'zero-shot-prompting', 'both']

        self.child_left: Operator = child_left
        self.child_right: Operator = child_left
        self.method: str = method
        self.embedding_comparison: str = embedding_comparison

        if self.method in ("threshold", "both"):
            assert em is not None
            assert threshold is not None
            self.em = em
            self.threshold = threshold
            self.embedding_method: str = embedding_method
            self.serialization_embedding: Callable[[dict], str] = serialization_embedding

            if self.embedding_comparison == "COLUMN_WISE":
                # store the embeddings for every colum in multiple vector indices
                assert columns_left is not None and columns_right is not None
                assert len(columns_left) == len(columns_right)

                # Init n Vector Index
                self.vector_indices = [vector_store_type(em.get_embedding_size()) for _ in range(len(columns_left))]
            else:
                # store the embeddings for every row in the vector index
                self.vector_index = vector_store_type(em.get_embedding_size()) # Init Vector Index


        if self.method in ("zero-shot-prompting", "both"):
            assert sv is not None
            self.sv = sv
            self.zs_system_prompt = zs_system_prompt
            self.zs_template = zs_template

            if serialization_zero_shot_prompting is None:
                self.serialization_zero_shot_prompting = self.default_serialization_zero_shot_prompting
            else:
                self.serialization_zero_shot_prompting = serialization_zero_shot_prompting

            if zs_template is None:
                self.zs_template = self.ZERO_SHOT_PROMPTING_TEMPLATE
            else:
                self.zs_template = zs_template

        # For COLUMN_WISE comparison -> shape: (n, #columns, embedding size)
        # For RECORD_WISE -> shape (n, embedding size)
        self.embeddings: np.array = None
        self.embeddings_map: dict[Any, np.array] = {}

        self.join_columns_left: None | list[str] | list[SQLColumn] = None
        self.join_columns_right: None | list[str] | list[SQLColumn] = None

        self.records_left: Optional[list[dict]] = []
        self.record_right: Optional[dict] = None

        self.distances: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.index_left: Optional[int] = None

        super().__init__(child_left, child_right, None)

        # Load left & right join columns for serialization with remapped column names
        #   (Uses column map build in Super-Class)
        # E.g. keys-left: [y], keys-right: [y] -> join_columns_left->[L.y], join_columns_right->[R.y]
        columns_left, columns_right = self._build_join_columns(columns_left, columns_right)
        self.join_columns_left: list[SQLColumn] = columns_left
        self.join_columns_right: list[SQLColumn] = columns_right


    def open(self) -> Operator:
        self.child_left.open()

        logging.debug("Loading records")
        # Iterate over left relation -> store remap records [x, y]->[x, L.y]
        self.records_left = [self._remap_record("left", row) for row in self.child_left]

        if self.method in ("threshold", "both") and len(self.join_columns_left) > 0:
            logging.debug("Embedding records")
            self.embeddings = self._create_left_embeddings() # Serialize and embedd left records

            logging.debug("Save to vector store")

            if self.embedding_comparison == "COLUMN_WISE":
                # Store the embeddings for all columns in their respective vector index
                for i, vi in enumerate(self.vector_indices):
                    vi.reset()
                    # noinspection PyArgumentList
                    vi.add(self.embeddings[:, i, :])
            else:
                self.vector_index.reset()
                # noinspection PyArgumentList
                self.vector_index.add(self.embeddings) # Insert left records in vector store

        self.child_left.close()
        self.child_right.open()
        return self

    def __next__(self) -> dict:
        if len(self.records_left) == 0: # No left records -> no matches
            self.close()
            raise StopIteration

        while True:
            if self.record_right is None:
                # Fix next right remapped ([y, z] -> [R.y, z]) record
                self.record_right = self._remap_record("right", next(self.child_right))

                # Embedd fixed (right) record
                if self.method in ("threshold", "both"):
                    if self.embedding_comparison == "COLUMN_WISE":
                        # Embedd and compare the cosine similarities for all join columns separately
                        # If mean cosine similarity (for all join columns) > threshold -> consider as match
                        keys_right = [str(self.record_right[col.column_name]) for col in self.join_columns_right]
                        embs_right = self.em(keys_right)

                        indices_total = set()
                        for i, vi in enumerate(self.vector_indices):
                            # Perform range query on vector Stores for every column embedding
                            # noinspection PyArgumentList
                            _, _, indices = vi.range_search(x=np.array([embs_right[i]]), thresh=self.threshold)
                            for idx in indices:
                                indices_total.add(idx)

                        self.indices = []
                        self.distances = []

                        # If any of the join columns is within the threshold,
                        #   then test if mean of all columns is > threshold
                        for idx in indices_total:
                            embs_left = self.embeddings[idx, :, :]
                            dist = [cosine_similarity(embs_left[i], embs_right[i]) for i in range(len(self.join_columns_right))]
                            avg_dist = np.mean(dist)
                            if avg_dist > self.threshold:
                                self.indices.append(idx)
                                self.distances.append(avg_dist)
                    else:
                        # Embedd and compare the pooled record
                        if self.embedding_method == "FULL_SERIALIZED":
                            # Embedd entire record
                            # LLM.embedd(serialize(record))
                            key = str({col.column_name: self.record_right[col.column_name] for col in self.join_columns_right})
                            embedding_right = self.em(key)
                        else:
                            # Embedd every column separately -> build average (Mean Pooling)
                            # AVG(LLM.embedd(serialize(record.column1)), LLM.embedd(serialize(record.column2)), ...)
                            key_elements = list({str(self.record_right[col.column_name]) for col in self.join_columns_right})
                            key_elements = list(km for km in key_elements if km not in self.embeddings_map)
                            if len(key_elements) > 0:
                                # Fills embedding map {key -> embedding}, so the same data is not embedded multiple times
                                for km, emb in zip(key_elements, self.em(key_elements)):
                                    self.embeddings_map[km] = emb

                            embedding_right = np.average(
                                [self.embeddings_map[str(self.record_right[col.column_name])] for col in self.join_columns_right],
                                axis=0)

                        # Perform range query on vector Store
                        # noinspection PyArgumentList
                        _, distances, indices = self.vector_index.range_search(x=np.array([embedding_right]), thresh=self.threshold)
                        self.indices = list(indices)
                        self.distances = list(distances)
                else:
                    # indices to left records within the range of threshold
                    self.indices = [i for i in range(len(self.records_left))]

                self.index_left = 0

            try:
                # Retrieve next left record from potential matches
                idx = self.indices[self.index_left]

                self.index_left += 1
                rec = self.records_left[idx] | self.record_right # Merge records

                debug_msg = f"Joined record {rec}"
                if self.method in ("threshold", "both"):
                    debug_msg += f" distance {self.distances[self.index_left-1]}"

                logging.debug( debug_msg)

                # Use LLM for entity matching ("Is record A: ... equal to record B: ...")
                if self.method in ("zero-shot-prompting", "both"):
                    # Reduce record to key columns
                    rec_a = {col.column_name: self.records_left[idx][col.column_name] for col in self.join_columns_left}
                    rec_b = {col.column_name: self.record_right[col.column_name] for col in self.join_columns_right}

                    # Fill template with serialized record
                    prompt = self.zs_template.format(
                        a = self.serialization_zero_shot_prompting(rec_a),
                        b = self.serialization_zero_shot_prompting(rec_b))

                    # entity matching not passing -> continue with next left record
                    if not self.sv(prompt, system_prompt=self.zs_system_prompt):
                        logging.debug(f"Prompt: \"{prompt}\" filed in semantic validation")
                        continue

                return rec

            # Left records exhausted -> continue with next right record
            except IndexError:
                self.record_right = None

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"⋈_{{{', '.join(map(str, self.join_columns_left))} ≈ {', '.join(map(str, self.join_columns_right))}}}"

    def _build_join_columns(self, columns_left, columns_right):
        # TODO: Soft Search For Join Columns
        join_columns_left: list[SQLColumn] = []
        join_columns_right: list[SQLColumn] = []

        # When no columns provided -> use entire record
        if columns_left is None:
            columns_left = [self.columns_map["left"][col.column_name] for col in self.child_left.table.table_structure]
        else:
            columns_left = [self.columns_map["left"][col] if col in self.columns_map["left"] else col for col in columns_left]

        if columns_right is None:
            columns_right = [self.columns_map["right"][col.column_name] for col in self.child_right.table.table_structure]
        else:
            columns_right = [self.columns_map["right"][col] if col in self.columns_map["right"] else col for col in columns_right]

        # search join columns from structure
        table_structure_map = {col.column_name: col for col in self.table.table_structure}

        missing_columns_left, missing_columns_right = [], []
        for col in columns_left:
            try:
                join_columns_left.append(table_structure_map[col])
            except KeyError:
                missing_columns_left.append(col)

        for col in columns_right:
            try:
                join_columns_right.append(table_structure_map[col])
            except KeyError:
                missing_columns_right.append(col)

        # Check availability of key-columns
        assert len(missing_columns_left) == 0, \
            f"Columns {missing_columns_left} not found for right relation {self.table.table_structure}"
        assert len(missing_columns_right) == 0, \
            f"Columns {missing_columns_right} not found for left relation {self.table.table_structure}"

        return join_columns_left, join_columns_right

    def _create_left_embeddings(self) -> np.array:
        # Embeds the records from left relation
        # For column wise comparison return embeddings for all columns,
        #   for record wise comparison, return embedding for pooled record

        if self.embedding_comparison == "COLUMN_WISE": # Returns array of shape (n, number columns, size embeddings)
            # Flatten Embeddings List -> shape (n * number columns)
            keys = [str(rec[col.column_name]) for rec in self.records_left for col in self.join_columns_left]
            embeddings = self.em(keys)
            len_cols = len(self.join_columns_left)
            # reshape to (n, number columns, size embeddings)
            return embeddings.reshape(embeddings.shape[0] // len_cols, len_cols, embeddings.shape[1])

        else: # Returns array of shape (n, size embeddings)
            if self.embedding_method == "FULL_SERIALIZED":
                # Embedd entire record
                # LLM.embedd(serialize(record))
                keys = [self.serialization_embedding({col.column_name: rec[col.column_name] for col in self.join_columns_left}) for rec in self.records_left]
                return self.em(keys)
            else:
                # Embedd every column separately -> build average
                # AVG(LLM.embedd(serialize(record.column1)), LLM.embedd(serialize(record.column2)), ...)
                key_elements_set = list({str(rec[col.column_name]) for col in self.join_columns_left for rec in self.records_left})
                self.embeddings_map = {key: embedding for key, embedding in zip(key_elements_set, self.em(key_elements_set))}
                key_elements_embeddings = np.array([
                    [self.embeddings_map[str(rec[col.column_name])] for col in self.join_columns_left]
                    for rec in self.records_left
                ])
                return np.average(key_elements_embeddings, axis=1)
