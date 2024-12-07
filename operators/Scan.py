from operators import Operator, Row, Col, DataType
from utils.DB import DBConnector
from utils.Model import EmbeddingModel, GenerationModel


class Scan(Operator):
    def __init__(self, name, embedding_model, generation_model, cosine_similarity_threshold=1):
        super().__init__()
        self.name = name
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.index = 0
        self.db_connector = DBConnector("../config.ini", use_vector=True)
        self.embedding_model: EmbeddingModel = embedding_model
        self.name_embedding = self.embedding_model.embedd(self.name)[0]
        # self.embedding_model.to_cpu()

        self.generation_model: GenerationModel = generation_model

        self.schema_name, self.table_name = self.get_table()
        self.columns = None

        if self.schema_name is not None:
            self.cursor = self.db_connector.get_cursor()
            self.cursor.execute(f"SELECT * FROM {self.schema_name}.{self.table_name} LIMIT 10")
            self.columns = [desc[0] for desc in self.cursor.description]
            # print(self.columns)

        # RAG
        self.pages = self.retrieve_documents()

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
                "SELECT schemaname, tablename FROM embeddings.tables ORDER BY embedding <=> %(embedding)s LIMIT 1",
                {"embedding": self.name_embedding})
            res = cursor.fetchone()
            return (res["schemaname"], res["tablename"]) if res is not None else (None, None)

    def retrieve_documents(self):
        with self.db_connector.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, content FROM embeddings.documents 
                WHERE (text_embedding <=> %(embedding)s) < %(threshold)s 
                ORDER BY text_embedding <=> %(embedding)s LIMIT 1
                """, {"embedding": self.name_embedding, "threshold": self.cosine_similarity_threshold})

            result = []

            for row in cursor.fetchall():
                # TODO: Validation & Tuple Extraction
                result.append(row)

        return result



if __name__ == '__main__':
    scan = Scan("actors", EmbeddingModel("../config.ini"), GenerationModel("../config.ini"))
    while curr_row := scan.next():
        # print(curr_row)
        pass
