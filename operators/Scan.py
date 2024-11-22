from operators import Operator, Row, Col, DataType
from utils.DB import DBConnector


class Scan(Operator):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.index = 0
        self.db_connector = DBConnector("../config.ini")
        self.schema_name, self.table_name = self.get_table()

        if self.schema_name is not None:
            self.cursor = self.db_connector.get_cursor()
            self.cursor.execute(f"SELECT * FROM {self.schema_name}.{self.table_name} LIMIT 10")
            self.columns = [desc[0] for desc in self.cursor.description]
            print(self.columns)

    def next(self) -> Row:
        row = Row(self.index)
        has_result = False

        if self.schema_name is not None:
            row_db = self.cursor.fetchone()
            has_result |= row_db is not None

            if row_db is not None:
                for key, value in row_db.items():
                    row.add_col(Col(key, DataType.TEXT, f"TABLE {self.schema_name}.{self.table_name}", value))

        self.index += 1

        return row if has_result else None

    def get_table(self) -> (str, str):
        with self.db_connector.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT schemaname, tablename FROM pg_tables
                WHERE 
                    schemaname NOT IN ('pg_catalog', 'information_schema', 'embeddings') AND 
                    similarity(tablename, %(name)s) > 0
                ORDER BY similarity(tablename, %(name)s) DESC
                LIMIT 1
                """, {"name": self.name})

            res = cursor.fetchone()

            if res is not None:
                return res["schemaname"], res["tablename"]
            else:
                return None, None


if __name__ == '__main__':
    scan = Scan("actors")
    while curr_row := scan.next():
        print(curr_row)
