import re

from typing import Any

class SQLColumn:
    PATTERN_FOREIGN_KEY = re.compile(r'FOREIGN_KEY\((.+)\.(.+)\.(.+)\)')

    def __init__(self, column_name, column_type, primary_key: bool = False, foreign_key: list[tuple[str, str, str]] = None, values: list[Any] = None):
        self.column_name = column_name
        self.column_type = column_type
        self.primary_key = primary_key
        self.foreign_key: list[tuple[str, str, str]] = foreign_key if foreign_key else []
        self.values: list[Any] = values if values else []

    def __str__(self) -> str:
        res = self.column_name + "(" + self.column_type
        res += ", PRIMARY_KEY" if self.primary_key else ""
        for fk in self.foreign_key:
            res += f", FOREIGN_KYE({fk[0]}.{fk[1]}.{fk[2]})"
        if len(self.values) > 0:
            res += f", VALUE_SAMPLES({', '.join(map(str, self.values))})"
        return res + ")"


class SQLTable:
    def __init__(self, table_schema: str | None, table_name: str, table_structure: list[SQLColumn]):
        self.table_schema: str | None = table_schema
        self.table_name: str = table_name
        self.table_structure: list[SQLColumn] = table_structure

    def __str__(self):
        if self.table_schema:
            return f"{self.table_schema}.{self.table_name}: {', '.join(map(str, self.table_structure))}"
        else:
            return f"{self.table_name}: {', '.join(map(str, self.table_structure))}"


class Column:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def __str__(self) -> str:
        return str(self.name)

    def get(self, t: dict) -> Any:
        return t[self.name]

class Constant:
    def __init__(self, value: Any) -> None:
        self.value: Any = value

    def __str__(self) -> str:
        return f"'{self.value}'" if isinstance(self.value, str) else str(self.value)

    def get(self, _: Any) -> Any:
        return self.value