from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

from utils.DB import DBConnector
from utils.Model import EmbeddingModel


def create_table_embeddings(vec_size=768):
    db = DBConnector("../config.ini", use_vector=True)
    m = EmbeddingModel("../config.ini")

    with db.get_cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings.tables (
                schemaname TEXT, tablename TEXT, embedding VECTOR(%(vec_size)s)
            );
            DELETE FROM embeddings.tables;
            """, {"vec_size": vec_size})

        cursor.execute("""SELECT schemaname, tablename FROM pg_tables 
            WHERE schemaname NOT IN ('pg_catalog', 'embeddings', 'information_schema')""")

        tables = cursor.fetchall()

        for batch in tqdm(DataLoader(Dataset.from_list(tables), batch_size=512)):
            embeddings = m.embedd(batch["tablename"])
            for schema, table, embedding in zip(batch["schemaname"], batch["tablename"], embeddings):
                cursor.execute(
                    "INSERT INTO embeddings.tables (schemaname, tablename, embedding) VALUES (%s, %s, %s)",
                    (schema, table, embedding))

    db.conn.commit()
    db.conn.close()

if __name__ == '__main__':
    create_table_embeddings()