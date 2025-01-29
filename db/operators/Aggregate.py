from typing import Any, Callable

from db.operators.Operator import Operator
from db.structure import SQLTable, Column, SQLColumn

from db.operators.Dummy import Dummy

from functools import reduce

class AggregationFunction:
    def __init__(self, column_name: str, result_type: Any,
                 function: Callable[[list[Any]], Any] | None = None ,
                 reduce_function: Callable[[Any, Any], Any] | None = None) -> None:
        self.column_name = column_name
        self.result_type = result_type
        assert function is not None or reduce_function is not None, "Provide either function or reduce_function"
        assert (function is None and reduce_function is not None) or (function is not None and reduce_function is None), "Provide either function or reduce_function"
        self.function = function
        self.reduce_function = reduce_function

class SumAggregation(AggregationFunction):
    def __init__(self, column_name: str):
        super().__init__(column_name, "Number", reduce_function=lambda result, value: result + value)

    def __str__(self):
        return f"SUM({self.column_name})"

class HashAggregate(Operator):
    def __init__(self, child_operator: Operator, columns: list[str], aggregation: list[AggregationFunction]):
        self.child_operator: Operator = child_operator
        self.aggregation = aggregation

        name = self.child_operator.table.table_name
        group_by_columns = [col for col in self.child_operator.table.table_structure if col.column_name in columns]
        self.group_by_columns_names = [col.column_name for col in group_by_columns]
        assert len(group_by_columns) == len(columns) # TODO: Add error MSG

        column_names_available = list(map(lambda col: col.column_name, self.child_operator.table.table_structure))
        aggregation_columns = [SQLColumn(col.column_name, col.result_type) for col in aggregation if col.column_name in column_names_available]
        self.aggregation_columns_names = [col.column_name for col in aggregation]
        assert len(aggregation_columns) == len(aggregation)

        self.map = {}
        self.iter = None

        super().__init__(SQLTable(None, name, group_by_columns + aggregation_columns), self.child_operator.num_tuples)

    def __next__(self) -> dict:
        key = next(self.iter)
        rows = self.map[key]
        record = {k: v for (k,v) in key}

        for aggregation in self.aggregation:
            if aggregation.reduce_function is not None:
                record[aggregation.column_name] = reduce(aggregation.reduce_function, map(lambda row: row[aggregation.column_name], rows))
            else:
                record[aggregation.column_name] = aggregation.function([row[aggregation.column_name] for row in rows])

        return record

    def __str__(self) -> str:
        return f"{self.get_description()} ({self.child_operator})"

    def open(self) -> None:
        self.child_operator.open()
        for row in self.child_operator:
            key = frozenset({k: v for k,v in row.items() if k in self.group_by_columns_names}.items())
            value = {k:v for k,v in row.items() if k in self.aggregation_columns_names}
            if key in self.map:
                self.map[key].append(value)
            else:
                self.map[key] = [value]
        self.child_operator.close()
        self.iter = iter(self.map)

    def next_vectorized(self) -> list[dict]:
        raise NotImplementedError()

    def close(self) -> None:
        self.iter.close()

    def get_description(self) -> str:
        return f"{','.join(self.group_by_columns_names)} ɣ_{{{','.join(map(str, self.aggregation))}}}"

    def get_structure(self) -> tuple[str, list] | str:
        return super().get_structure(), [self.child_operator.get_structure()]


