from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

from db.db import DBConnector
from models.embedding.Generic import GenericEmbeddingModel


def create_table_embeddings(vec_size=768):
    db = DBConnector("../config.ini", use_vector=True)
    m = GenericEmbeddingModel()

    with db.get_cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings.tables (
                schemaname TEXT, tablename TEXT, embedding VECTOR(%(vec_size)s)
            );
            DELETE FROM embeddings.tables;
            """, {"vec_size": vec_size})

        cursor.execute("""SELECT schemaname, tablename, CONCAT(schemaname, '.', tablename) AS name FROM pg_tables 
            WHERE schemaname NOT IN ('pg_catalog', 'embeddings', 'information_schema')""")

        tables = cursor.fetchall()

        for batch in tqdm(DataLoader(Dataset.from_list(tables), batch_size=512)):
            embeddings = m.embedd(batch["name"])
            for schema, table, embedding in zip(batch["schemaname"], batch["tablename"], embeddings):
                cursor.execute(
                    "INSERT INTO embeddings.tables (schemaname, tablename, embedding) VALUES (%s, %s, %s)",
                    (schema, table, embedding))

    db.conn.commit()
    db.conn.close()

if __name__ == '__main__':
    create_table_embeddings()