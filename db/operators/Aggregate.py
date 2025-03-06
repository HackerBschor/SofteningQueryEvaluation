import logging
from abc import ABC
from typing import Any, Callable, Type, Literal

import faiss
import numpy as np
import pandas as pd

from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.manifold import TSNE

import umap

import seaborn as sns

from db.operators.Operator import Operator
from db.structure import SQLTable, SQLColumn

from functools import reduce

from models.embedding.Model import EmbeddingModel
from models.text_generation.Model import TextGenerationModel


class AggregationFunction(ABC):
    """
    Abstract class for an aggregation function, that has to be passed to the Aggregation Operator.
    This class is used to apply an aggregation function to all elements in a group.
    They correspond to SQL functions such as "COUNT, SUM, MIN, MAX, ..."

    The concrete functions are implemented in the classes extending this abstract class.

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
        result_type: What is the type of the aggregated value (Numeric for SUM, String for STRING_AGG, ...)
        function (Callable[[list[Any]], Any]): The function that will be applied on the list of value
        reduce_function (Callable[[Any, Any], Any]): Alternatively to the function, a reduce function
        default_value: Start value for the reduce function
    """

    def __init__(self, aggregation_column: str, new_column_name: str, result_type: Any,
                 function: Callable[[list[Any]], Any] | None = None,
                 reduce_function: Callable[[Any, Any], Any] | None = None,
                 default_value: Any = None) -> None:
        self.aggregation_column = aggregation_column
        self.new_column_name = new_column_name
        self.result_type = result_type

        assert function is not None or reduce_function is not None, "Provide either function or reduce_function"
        assert ((function is None and reduce_function is not None) or
                (function is not None and reduce_function is None)), "Provide either function or reduce_function"

        self.function = function
        self.reduce_function = reduce_function
        self.default_value = default_value


class SumAggregation(AggregationFunction):
    """
        SUM Aggregation function ("SUM(column)" in SQL)

        Attributes:
            aggregation_column (str): The column that is aggregated
            new_column_name (str): The column in which the aggregated value is stored for the final record
        """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: result + value)

    def __str__(self):
        return f"SUM({self.aggregation_column})"


class MaxAggregation(AggregationFunction):
    """
    MAX Aggregation function ("MAX(column)" in SQL)

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: max(result, value))

    def __str__(self):
        return f"MAX({self.aggregation_column})"


class MinAggregation(AggregationFunction):
    """
    MIN Aggregation function ("MIN(column)" in SQL)

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: min(result, value))

    def __str__(self):
        return f"MIN({self.aggregation_column})"


class CountAggregation(AggregationFunction):
    """
    COUNT Aggregation function ("COUNT(column)" in SQL)

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: result + 1, default_value=0)

    def __str__(self):
        return f"COUNT({self.aggregation_column})"


class CountDistinctAggregation(AggregationFunction):
    """
    Distinct COUNT Aggregation function ("COUNT(Distinct column)" in SQL)

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        self.data = set()

        def rf(result, value):
            if value in self.data:
                return result

            self.data.update(value)
            return result + 1

        super().__init__(aggregation_column, new_column_name, "Number", reduce_function=rf, default_value=0)

    def __str__(self):
        return f"COUNT-DISTINCT({self.aggregation_column})"


class AvgAggregation(AggregationFunction):
    """
    AVG Aggregation function ("AVG(Distinct column)" in SQL)

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number", function=lambda rows: np.average(rows))

    def __str__(self):
        return f"AVG({self.aggregation_column})"


class StringAggregation(AggregationFunction):
    """
    Aggregation of values by string concatenation seperated by a delimiter

    E.g. ["a", "b", "c"] + delimiter=", " -> "a, b, c"

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
        delimiter (str): The delimiter for the strings aggregated
    """
    def __init__(self, aggregation_column: str, new_column_name: str, delimiter: str = ", "):
        self.delimiter = delimiter
        super().__init__(aggregation_column, new_column_name, "Text", function=lambda rows: delimiter.join(rows))

    def __str__(self):
        return f"STRING_AGG({self.aggregation_column}, \"{self.delimiter}\")"


class SetAggregation(AggregationFunction):
    """
    Aggregation of values in a set

    E.g. ["a", "a", "b", "b, "c""] -> {"a", "b", "c"}

    Attributes:
        aggregation_column (str): The column that is aggregated
        new_column_name (str): The column in which the aggregated value is stored for the final record
        delimiter (str): The delimiter for the strings aggregated
    """
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "set", function=lambda rows: set(rows))

    def __str__(self):
        return f"SET({self.aggregation_column})"


class Aggregate(Operator):
    """
    Abstract class of the Aggregation Operator (GROUP BY in SQL)

    Aggregates values by key values and applies an aggregation function to the values ion the same bucket.
    E.g. [{"key": 1, "value": 10}, {"key": 1, "value": 5}, {"key": 2, "value": 9}]
    Aggregate by key, sum value -> [{"key": 1, value: 15}, {"key": 2, value: 9}]

    Attributes:
        child_operator (Operator): The operator that generates the records for aggregation
        columns (list[str]): Key columns for aggregation
        aggregation (list[AggregationFunction]): Aggregation functions that are applied to list of recrods with same key
    """
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction]):
        self.child_operator: Operator = child_operator
        self.aggregation: list[AggregationFunction] = aggregation

        name = self.child_operator.table.table_name

        # Check that all group by values are present
        group_by_columns = [col for col in self.child_operator.table.table_structure if col.column_name in columns]
        self.group_by_columns_names = [col.column_name for col in group_by_columns]
        assert len(group_by_columns) == len(columns)  # TODO: Add error MSG

        # All columns provided by the structure of child operator
        column_names_available = list(map(lambda col: col.column_name, self.child_operator.table.table_structure))

        # Check that all columns required for aggregation are present
        aggregation_columns = [
            SQLColumn(col.new_column_name, col.result_type) for col in aggregation
            if col.aggregation_column in column_names_available]

        self.aggregation_columns_names = [col.aggregation_column for col in aggregation]
        assert len(aggregation_columns) == len(aggregation)

        super().__init__(SQLTable(None, name, group_by_columns + aggregation_columns), self.child_operator.num_tuples)

    def __next__(self) -> dict:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        raise NotImplementedError()

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def get_description(self) -> str:
        return f"{','.join(self.group_by_columns_names)} ɣ_{{{','.join(map(str, self.aggregation))}}}"

    def get_structure(self) -> tuple[str, list] | str:
        return super().get_structure(), [self.child_operator.get_structure()]

    def _crate_record(self, key: dict, rows: list[dict]) -> dict:
        """
        Builds the final record for a "bucket" of aggregated values
        E.g. GroupBy k1,k2, Sum(v1), STRING_AGG(v2, delimiter=', ')
        key = {"k1": 1, "k2": "a"}, rows = [{"v1": 10, "v2": "hallo"}, {"v1": 5, "v2": "welt"}]
            -> {"k1": 1, "k2": "a", v1: 15, v2: "hallo, welt"}
        """

        record = key # the key is always part of result

        for aggregation in self.aggregation: # apply all aggregation functions
            # remap records to list of values
            aggregated_values = [row[aggregation.aggregation_column] for row in rows]

            if aggregation.reduce_function is not None:
                # Applies reduce function (with start value, if present)
                if aggregation.default_value is not None:
                    record[aggregation.new_column_name] = reduce(
                        aggregation.reduce_function, aggregated_values, aggregation.default_value)
                else:
                    record[aggregation.new_column_name] = reduce(aggregation.reduce_function, aggregated_values)
            else:
                # Applies function to value list
                record[aggregation.new_column_name] = aggregation.function(aggregated_values)

        return record


class HashAggregate(Aggregate):
    """
    Implements a HashAggregate (DEFAULT)

    Iterates over input relation and aggregate the records (according to the key values) in a hash table.
    Then, iterate over the keys in the hash table. Apply the aggregation functions on the record list and return
     the key with the applied aggregation.

    Attributes:
        child_operator (Operator): The operator that generates the records for aggregation
        columns (list[str]): Key columns for aggregation
        aggregation (list[AggregationFunction]): Aggregation functions that are applied to list of recrods with same key
    """
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction]):
        self.map = {}
        self.iter = None

        super().__init__(child_operator, columns, aggregation)

    def __next__(self) -> dict:
        key = next(self.iter) #Get next key form HashMap
        # apply aggregation functions, return merged (key + aggregation) record
        return self._crate_record({k: v for (k, v) in key}, self.map[key])

    def open(self) -> Operator:
        self.map = {}
        self.iter = None

        self.child_operator.open()
        for row in self.child_operator:
            # Reduce record to key values -> create frozen set for hashing
            key = frozenset({k: v for k, v in row.items() if k in self.group_by_columns_names}.items())

            # Collect the values required for the aggregation functions in the HashMap
            value = {k: v for k, v in row.items() if k in self.aggregation_columns_names}

            if key in self.map:
                self.map[key].append(value)
            else:
                self.map[key] = [value]

        self.child_operator.close()
        self.iter = iter(self.map)
        return self

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.map = {}
        self.iter = None
        self.child_operator.close()


class SoftAggregate(Aggregate):
    @staticmethod
    def default_serialization(x: dict) -> str:
        return '{' + ', '.join([f'{k.split(".")[-1]}: \'{v}\'' for k, v in sorted(x.items(), key=lambda x: x[0]) if v is not None]) + '}'

    # TODO: Prompt Tuning
    KEY_SUMMARY_SYSTEM_PROMPT: str = "Create a summary for the record input. " +\
        "E.g. for Data: [{'name': 'Calico'}, {'name': 'Google LLC.'}, {'name': 'Nest Labs'}] " +\
        "return 'Alphabet Inc.'\n\n" +\
        "Answer with one single summary!"

    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction],
                 em: EmbeddingModel, create_key_summary: bool = False, tgm: TextGenerationModel = None,
                 key_summary_sp: str = KEY_SUMMARY_SYSTEM_PROMPT,
                 serialization: Callable[[dict], str] = None):
        self.clusters: dict = {}
        self.iter: iter = None

        self.em: EmbeddingModel = em

        self.serialization = self.default_serialization if serialization is None else serialization

        self.keys: list[dict] = []
        self.rows: list[dict] = []
        self.embeddings: np.array = None

        self.create_key_summary: bool = create_key_summary
        self.tgm: TextGenerationModel | None = tgm
        self.key_summary_sp: str = key_summary_sp

        if self.create_key_summary:
            new_table_structure = [SQLColumn("key", "Text")] + \
                [col for col in self.table.table_structure if col.column_name in self.aggregation_columns_names]
            self.table.structure = new_table_structure

        super().__init__(child_operator, columns, aggregation)

    def open(self) -> Operator:
        self.child_operator.open()

        for row in self.child_operator:
            key = {k: v for k, v in row.items() if k in self.group_by_columns_names}
            value = {k: v for k, v in row.items() if k in self.aggregation_columns_names}
            embedding = self.em(str(key))  # TODO: Check if string always ordered

            self.keys.append(key)
            self.rows.append(value)
            self.embeddings.append(embedding)

        self.child_operator.close()

        self._build_clusters_map(np.array(self.embeddings))
        self.iter = iter(self.clusters.keys())
        return self

    def __next__(self) -> dict:
        cluster: list[int] = self.clusters[next(self.iter)]

        if self.create_key_summary:
            prompt = "Data: " + ", ".join([str(self.keys[idx]) for idx in cluster])
            logging.debug(prompt)
            key = {"key": self.tgm(prompt, self.key_summary_sp)}
        else:
            key = {}
            for name in self.group_by_columns_names:
                key[name] = {self.keys[idx][name] for idx in cluster}

        rows = [self.rows[idx] for idx in cluster]
        return self._crate_record(key, rows)

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.iter.close()

    def _build_clusters_map(self, embeddings):
        raise NotImplementedError()

    def visualize(self):
        clusters = []
        rows = []
        for cluster, embs in self.clusters.items():
            clusters.append(pd.Series([str(self.keys[cluster]) for _ in range(len(embs))]))
            rows.append(self.embeddings[embs])

        clusters = pd.concat(clusters)
        rows = np.concatenate(rows)

        tsne = TSNE(n_components=2, perplexity=min(15, len(rows)-1), random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(rows)
        df = pd.DataFrame({ "x": vis_dims[:, 0], "y": vis_dims[:, 1], "c": clusters })
        sns.lmplot(x='x', y='y', data=df, hue='c', fit_reg=False)




class SoftAggregateFaissKMeans(SoftAggregate):
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction],
                 em: EmbeddingModel, num_clusters: int | None = None, num_clusters_relative: float | None = None,
                 **kwargs):

        self.num_clusters: int | None = num_clusters
        self.num_clusters_relative: float | None = num_clusters_relative
        self.centroids: np.array = None

        assert sum(x is not None for x in [self.num_clusters, self.num_clusters_relative]) > 0, \
            "Set num_clusters or num_clusters_relative"
        assert sum(x is not None for x in [self.num_clusters, self.num_clusters_relative]) == 1, \
            "Set only num_clusters or num_clusters_relative"
        if self.num_clusters_relative is not None:
            assert 0 < self.num_clusters_relative < 1

        super().__init__(child_operator, columns, aggregation, em, **kwargs)

    def _build_clusters_map(self, embeddings: np.array):
        if self.num_clusters_relative:
            self.num_clusters = int(self.num_clusters_relative * float(len(self.rows)))

        kmeans = faiss.Kmeans(d=self.em.get_embedding_size(), k=self.num_clusters, niter=20, verbose=True)
        kmeans.train(embeddings)

        # 1-NN search to find nearest cluster center
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()

        self.centroids = kmeans.centroids

        self.clusters = {i: [] for i in range(self.num_clusters)}

        for idx, label in enumerate(labels):
            self.clusters[label].append(idx)


class SoftAggregateScikit(SoftAggregate):
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction],
                 em: EmbeddingModel,
                 cluster_class: Type[ClusterMixin and BaseEstimator], cluster_params: dict,
                 serialization_mode: Literal["FULL_SERIALIZED", "FIELD_SERIALIZED"] = "FULL_SERIALIZED",
                 reduce_dimensions: int | None = 0,
                 serialization = None):

        self.clustering = cluster_class(**cluster_params)
        self.serialization_mode = serialization_mode
        self.reduce_dimensions = reduce_dimensions

        super().__init__(child_operator, columns, aggregation, em, serialization=serialization)

    def open(self) -> Operator:
        self.child_operator.open()

        embeddings = []
        for row in self.child_operator:
            key = {k: v for k, v in row.items() if k in self.group_by_columns_names}
            value = {k: v for k, v in row.items() if k in self.aggregation_columns_names}

            if self.serialization_mode == "FULL_SERIALIZED":
                embedding = self.em(self.serialization(key))
            else:
                embedding = np.concatenate(self.em([str(v) for v in key.values()]))

            self.keys.append(key)
            self.rows.append(value)
            embeddings.append(embedding)

        self.embeddings = np.array(embeddings)

        if self.reduce_dimensions is not None:
            umap_model = umap.UMAP(n_components=self.reduce_dimensions)
            self.embeddings = umap_model.fit_transform(self.embeddings)

        self.child_operator.close()

        self.clustering.fit(self.embeddings)

        unclustered_elements = 0
        for i, x in enumerate(self.clustering.labels_):
            if x == -1:
                self.clusters[f"unclustered-{unclustered_elements}"] = [i]
                unclustered_elements += 1
            else:
                if x in self.clusters:
                    self.clusters[x].append(i)
                else:
                    self.clusters[x] = [i]

        self.iter = iter(self.clusters.keys())
        return self
    