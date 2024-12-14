from operators import Operator
from utils.Model import EmbeddingModel, GenerationModel


class Scan(Operator):
    def __init__(self, name, db_connector, embedding_model, generation_model, cosine_similarity_threshold=1):
        self.name = name
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.db_connector = db_connector
        self.embedding_model: EmbeddingModel = embedding_model
        self.name_embedding = self.embedding_model.embedd(self.name)[0]
        self.generation_model: GenerationModel = generation_model

        self.schema_name, self.table_name = self.get_table()

        assert self.schema_name is not None, f"No table found for '{name}'"

        self.cursor = self.db_connector.get_cursor()
        self.cursor.execute(f"SELECT * FROM {self.schema_name}.{self.table_name}")

        super().__init__(name, [desc[0] for desc in self.cursor.description])

    def __str__(self):
        return self.name

    def get_table(self) -> (str, str):
        with self.db_connector.get_cursor() as cursor:
            cursor.execute(
                "SELECT schemaname, tablename FROM embeddings.tables ORDER BY embedding <=> %(embedding)s LIMIT 1",
                {"embedding": self.name_embedding})
            res = cursor.fetchone()
            return (res["schemaname"], res["tablename"]) if res is not None else (None, None)


    def __next__(self) -> dict:
        return next(self.cursor)
