import logging

import psycopg2
import psycopg2.extras

import configparser

from pgvector.psycopg2 import register_vector

from db.structure import SQLTable, SQLColumn
from utils import get_config


class DBConnector:
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

    SQL_GET_N_DISTINCT_VALUES = "SELECT DISTINCT {} FROM {}.{} TABLESAMPLE BERNOULLI(1) LIMIT {}"

    def __init__(self, config: str, use_vector: bool = False, load_db: bool = False, num_distinct_value: int = 10):
        config: configparser.ConfigParser = get_config(config)

        self.num_distinct_value = num_distinct_value

        logging.debug("Connecting to database")

        self.conn = psycopg2.connect(
            database=config["DB"].get("database"),
            host=config["DB"].get("host"),
            user=config["DB"].get("user"),
            password=config["DB"].get("password"),
            port=config["DB"].get("port"))

        if not self.conn:
            raise Exception("Could not connect to database.")

        self.conn.autocommit = False

        if use_vector:
            register_vector(self.conn)

        if load_db:
            self.tables: dict[str, SQLTable] = self.load_table_structure()

    def get_cursor(self) -> psycopg2.extensions.cursor:
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    @staticmethod
    def parse_sql_column(descr) -> SQLColumn:
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

    def parse_table(self, row):
        table_schema = row["table_schema"]
        table_name = row["table_name"]
        structure = []

        logging.debug(f"Load table {table_schema}.{table_name}")

        for col in row["table_structure"].split(", "):
            sql_column = self.parse_sql_column(col)

            if "json" not in sql_column.column_type:
                with self.get_cursor() as cursor:
                    cursor.execute(self.SQL_GET_N_DISTINCT_VALUES.format(
                        sql_column.column_name, table_schema, table_name, self.num_distinct_value))

                    sql_column.values = [v[sql_column.column_name] for v in cursor.fetchall()]

            structure.append(sql_column)

        return SQLTable(table_schema, table_name, structure)


    def load_table_structure(self) -> dict[str, SQLTable]:
        with self.get_cursor() as cursor:
            cursor.execute(self.SQL_FETCH_TABLES)
            return {f"{row['table_schema']}.{row['table_name']}": self.parse_table(row) for row in cursor.fetchall()}

if __name__ == '__main__':
    DBConnector(config='../config.ini')
