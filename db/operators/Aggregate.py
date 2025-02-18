import logging
from typing import Any, Callable, Type

import faiss
import numpy as np
import pandas as pd

from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.manifold import TSNE

import seaborn as sns

from db.operators.Operator import Operator
from db.structure import SQLTable, SQLColumn

from functools import reduce

from models.embedding.Model import EmbeddingModel
from models.text_generation.Model import TextGenerationModel


class AggregationFunction:
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
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: result + value)

    def __str__(self):
        return f"SUM({self.aggregation_column})"


class MaxAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: max(result, value))

    def __str__(self):
        return f"MAX({self.aggregation_column})"


class MinAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: min(result, value))

    def __str__(self):
        return f"MIN({self.aggregation_column})"


class CountAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number",
                         reduce_function=lambda result, value: result + 1, default_value=0)

    def __str__(self):
        return f"COUNT({self.aggregation_column})"


class CountDistinctAggregation(AggregationFunction):
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
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "Number", function=lambda rows: np.average(rows))

    def __str__(self):
        return f"AVG({self.aggregation_column})"


class StringAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str, delimiter: str = ", "):
        self.delimiter = delimiter
        super().__init__(aggregation_column, new_column_name, "Text", function=lambda rows: delimiter.join(rows))

    def __str__(self):
        return f"STRING_AGG({self.aggregation_column}, \"{self.delimiter}\")"


class DistinctAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str, column_type: str):
        super().__init__(aggregation_column, new_column_name, column_type,
                         function=lambda rows: rows[0], default_value=0)

    def __str__(self):
        return f"DISTINCT({self.aggregation_column})"


class SetAggregation(AggregationFunction):
    def __init__(self, aggregation_column: str, new_column_name: str):
        super().__init__(aggregation_column, new_column_name, "set", function=lambda rows: set(rows))

    def __str__(self):
        return f"LIST({self.aggregation_column})"


class Aggregate(Operator):
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction]):
        self.child_operator: Operator = child_operator
        self.aggregation: list[AggregationFunction] = aggregation

        name = self.child_operator.table.table_name
        group_by_columns = [col for col in self.child_operator.table.table_structure if col.column_name in columns]
        self.group_by_columns_names = [col.column_name for col in group_by_columns]
        assert len(group_by_columns) == len(columns)  # TODO: Add error MSG

        column_names_available = list(map(lambda col: col.column_name, self.child_operator.table.table_structure))

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
        record = key
        for aggregation in self.aggregation:
            if aggregation.reduce_function is not None:
                reduce_params = [
                    aggregation.reduce_function, map(lambda row: row[aggregation.aggregation_column], rows)]

                if aggregation.default_value is not None:
                    reduce_params.append(aggregation.default_value)
                record[aggregation.new_column_name] = reduce(*reduce_params)
            else:
                record[aggregation.new_column_name] = aggregation.function(
                    [row[aggregation.aggregation_column] for row in rows])

        return record


class HashAggregate(Aggregate):
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction]):
        self.map = {}
        self.iter = None

        super().__init__(child_operator, columns, aggregation)

    def __next__(self) -> dict:
        key = next(self.iter)
        return self._crate_record({k: v for (k, v) in key}, self.map[key])

    def open(self) -> None:
        self.map = {}
        self.iter = None

        self.child_operator.open()
        for row in self.child_operator:
            key = frozenset({k: v for k, v in row.items() if k in self.group_by_columns_names}.items())
            value = {k: v for k, v in row.items() if k in self.aggregation_columns_names}
            if key in self.map:
                self.map[key].append(value)
            else:
                self.map[key] = [value]
        self.child_operator.close()
        self.iter = iter(self.map)

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.map = {}
        self.iter = None
        self.child_operator.close()


class SoftAggregate(Aggregate):
    # TODO: Prompt Tuning
    KEY_SUMMARY_SYSTEM_PROMPT: str = "Create a summary for the record input. " +\
        "E.g. for Data: [{'name': 'Calico'}, {'name': 'Google LLC.'}, {'name': 'Nest Labs'}] " +\
        "return 'Alphabet Inc.'\n\n" +\
        "Answer with one single summary!"

    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction],
                 em: EmbeddingModel, create_key_summary: bool = False, tgm: TextGenerationModel = None,
                 key_summary_sp: str = KEY_SUMMARY_SYSTEM_PROMPT):
        self.clusters: dict = {}
        self.iter: iter = None

        self.em: EmbeddingModel = em

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

    def open(self) -> None:
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

        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
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
                 em: EmbeddingModel, cluster_class: Type[ClusterMixin and BaseEstimator], cluster_params: dict,
                 **kwargs):

        self.clustering = cluster_class(**cluster_params)

        super().__init__(child_operator, columns, aggregation, em, **kwargs)

    def open(self) -> None:
        self.child_operator.open()

        embeddings = []
        for row in self.child_operator:
            key = {k: v for k, v in row.items() if k in self.group_by_columns_names}
            value = {k: v for k, v in row.items() if k in self.aggregation_columns_names}
            embedding = self.em(str(key)) # TODO: Check if string always ordered

            self.keys.append(key)
            self.rows.append(value)
            embeddings.append(embedding)

        self.embeddings = np.array(embeddings)

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
