import re

from operators.Scan import Scan


class Parser:
    pattern = r"SELECT (.+) FROM ([a-zA-Z0-9_, ]+) WHERE (.+)?"

    def __init__(self):
        pass

    @staticmethod
    def parse(query: str):
        match = re.match(Parser.pattern, query)
        if match:
            tables = Parser.parse_from(match.group(2))
            Parser.parse_select(match.group(1), tables)
            Parser.parse_where(match.group(3))

    @staticmethod
    def parse_select(s: str, tables: dict[str, Scan]):
        pass


    @staticmethod
    def parse_from(s: str) -> dict[str, Scan]:
        tables =  {}

        for table_key in s.split(","):
            table_key = table_key.strip()
            if " " not in table_key:
                table, key = table_key, table_key
            else:
                table, key = table_key.strip().split(" ")
            tables[key] = Scan(table, None, None)

        return tables



    @staticmethod
    def parse_where(s: str):
        print(s)



if __name__ == '__main__':
    print(Parser.parse("SELECT a.x FROM actors a, test b WHERE a.x = b.y"))