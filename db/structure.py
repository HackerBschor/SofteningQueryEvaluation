import re

from typing import Any

class SQLColumn:
    PATTERN_FOREIGN_KEY = re.compile(r'FOREIGN_KEY\((.+)\.(.+)\.(.+)\)')

    def __init__(self, column_name, column_type, primary_key: bool = False, foreign_key: list[tuple[str, str, str]] = ()):
        self.column_name = column_name
        self.column_type = column_type
        self.primary_key = primary_key
        self.foreign_key: list[tuple[str, str, str]] = foreign_key

    @staticmethod
    def parse(descr) -> 'SQLColumn':
        descr_enc = descr.split(":")
        if len(descr_enc) < 2:
            raise Exception(f"Invalid column description: {descr}")
        elif len(descr_enc) == 2:
            return SQLColumn(descr_enc[0], descr_enc[1])
        else:
            col = SQLColumn(descr_enc[0], descr_enc[1])

            for attr in descr_enc[2:]:
                if attr == "PRIMARY_KEY":
                    col.primary_key = True
                if attr.startswith("FOREIGN_KEY"):
                    match = SQLColumn.PATTERN_FOREIGN_KEY.match(attr)
                    if match:
                        (s, t, c) = match.groups()
                        col.foreign_key.append((s, t, c))

            return col

    def __str__(self) -> str:
        res = self.column_name + "(" + self.column_type
        res += ", PRIMARY_KEY" if self.primary_key else ""
        for fk in self.foreign_key:
            res += f", FOREIGN_KYE({fk[0]}.{fk[1]}.{fk[2]})"
        return res + ")"


class SQLTable:
    def __init__(self, table_schema: str | None, table_name: str, table_structure: str | list[SQLColumn]):
        self.table_schema: str | None = table_schema
        self.table_name: str = table_name
        if isinstance(table_structure, str):
            self.table_structure: list[SQLColumn] = self._parse_structure(table_structure)
        else:
            self.table_structure = table_structure

    def __str__(self):
        if self.table_schema:
            return f"{self.table_schema}.{self.table_name}: {', '.join(map(str, self.table_structure))}"
        else:
            return f"{self.table_name}: {', '.join(map(str, self.table_structure))}"

    @staticmethod
    def _parse_structure(table_structure: str) -> list[SQLColumn]:
        return [SQLColumn.parse(col) for col in table_structure.split(", ")]


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