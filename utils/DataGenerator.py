import requests
import re
import bz2
import os

from typing import List
from tqdm import tqdm

from xml.etree import ElementTree

from torch.utils.data import DataLoader
from datasets import Dataset

from utils.Model import EmbeddingModel
from utils.DB import DBConnector


class WikiDataGenerator:
    _pattern_url = r'<a href="(enwiki-latest-pages-articles-multistream<n>\.xml-p[^.]+\.bz2)">'
    _pattern_file = re.compile(r'(enwiki-latest-pages-articles-multistream\d+\.xml)-p[^.]+\.bz2')

    @staticmethod
    def get_wikipedia_data(
            base_url: str = "https://dumps.wikimedia.org/enwiki/latest/",
            path: str = "../data/wikipedia/",  max_files: int = 5):

        response = requests.get(base_url)
        html_string = response.text
        i = 0

        while i < max_files:
            match = re.search(WikiDataGenerator._pattern_url.replace("<n>", str(i + 1)), html_string)

            if not match:
                break

            WikiDataGenerator.download_file(path, base_url+match.group(1))

            i += 1

    @staticmethod
    def download_file(path: str, url: str):
        print("Downloading file", url, "to", path)

        local_filename: str = os.path.join(path, url.split('/')[-1])

        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename

    @staticmethod
    def decompress_wiki_data(path_src: str = "../data/wikipedia/", path_dest: str = "../data/wikipedia/decompressed"):
        assert os.path.exists(path_src), f"Source path '{path_src}' does not exist"
        assert os.path.exists(path_dest), f"Destination path '{path_dest}' does not exist"

        for filename in os.listdir(path_src):
            match = WikiDataGenerator._pattern_file.match(filename)

            if not match:
                continue

            file_src = os.path.join(path_src, filename)
            file_dest = os.path.join(path_dest, match.group(1))

            print("Decompressing file", file_src, "to", file_dest)

            with open(file_dest, 'wb') as new_file, bz2.BZ2File(file_src, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)

    @staticmethod
    def parse_wiki_file(file: str) -> List[dict]:
        xml_data = ElementTree.parse(file).getroot()
        namespaces = {"wiki": re.compile(r'{(.+)}').match(xml_data.tag).group(1)}

        pages = []
        for page in tqdm(xml_data.findall(f"wiki:page", namespaces)):
            revision = page.find("wiki:revision", namespaces)
            pages.append({
                "title": page.find("wiki:title", namespaces).text,
                "revision_id": revision.find("wiki:id", namespaces).text,
                "text": revision.find("wiki:text", namespaces).text})

        return pages

    @staticmethod
    def get_wikipedia_link(title, revision_id):
        return f"https://en.wikipedia.org/w/index.php?title={title}&oldid={revision_id}"


class VectorDB:
    similarities = {
        "L2Distance": "vector_l2_ops",
        "InnerProduct": "vector_ip_ops",
        "CosineDistance": "vector_cosine_ops",
        "L1Distance": "vector_l1_ops",
        "HammingDistance": "bit_hamming_ops",
        "JaccardDistance": "bit_jaccard_ops"
    }

    def __init__(self, config: str):
        self.db_connector = DBConnector(config, use_vector=True)

    def create_documents_vector_db(self, vec_size: int, reset: bool = False):
        with self.db_connector.get_cursor() as cur:
            if reset:
                print("Resetting vector database")
                cur.execute("""
                    DROP TABLE IF EXISTS public.documents;
                    DROP TABLE IF EXISTS public.embeddings;
                """)
                self.db_connector.conn.commit()

            print("Creating vector database")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings.documents (
                    id bigserial primary key,
                    revision_id bigint,
                    title text,
                    content text,
                    title_embedding vector(%(vec_size)s),
                    text_embedding vector(%(vec_size)s)
                );
                
                CREATE TABLE IF NOT EXISTS embeddings.embeddings (
                    id bigint , 
                    chunk_id bigint,
                    content text,
                    embedding vector(%(vec_size)s),
                    PRIMARY KEY (id, chunk_id)
                );
            """, {"vec_size": vec_size})

        self.db_connector.conn.commit()

    def create_index(self, method):
        assert method in self.similarities
        method = self.similarities[method]

        with self.db_connector.get_cursor() as cur:
            cur.execute(f"""
                CREATE INDEX ON embeddings.documents USING hnsw (text_embedding {method});
                CREATE INDEX ON embeddings.documents USING hnsw (title_embedding {method});
            """)

        self.db_connector.conn.commit()

    def insert_document(self, revision_id, title, text, title_embedding, text_embedding):
        with self.db_connector.get_cursor() as cur:
            cur.execute(
                """
                    INSERT INTO embeddings.documents (revision_id, title, content, title_embedding, text_embedding) 
                    VALUES (%s, %s, %s, %s, %s)
                    """, (revision_id, title, text, title_embedding, text_embedding))

        self.db_connector.conn.commit()


def process_wiki_data(path="../data/wikipedia/decompressed"):
    model = EmbeddingModel("../config.ini")
    vector_db = VectorDB("../config.ini")

    vector_db.create_documents_vector_db(vec_size=768, reset=True)

    for filename in os.listdir(path):
        print("Reading file", filename)
        file_path = os.path.join(path, filename)
        data = WikiDataGenerator.parse_wiki_file(file_path)

        if data[0]["title"] == "AccessibleComputing":
            data = data[1:]

        for batch in tqdm(DataLoader(Dataset.from_list(data), batch_size=512)):
            try:
                title_embeddings = model.embedd(batch["title"])
                text_embeddings = model.embedd(batch["text"])
            except Exception as e:
                with open("../logs/errors.log", "a", encoding="utf-8") as log:
                    log.write(str(e) + "\n" + str(batch))
                    log.write("\n\n")

                continue

            for row in zip(batch["revision_id"], batch["title"], batch["text"], title_embeddings, text_embeddings):
                vector_db.insert_document(*row)

        print("Removing file", filename)
        os.remove(file_path)


def test_embeddings(query, column):
    assert column in ["text", "title"], "Column must be title/ text"

    db = DBConnector("../config.ini", use_vector=True)
    m = EmbeddingModel("../config.ini")

    embedding = m.embedd(query)[0]

    with db.get_cursor() as cur:
        cur.execute(
            f"""
                SELECT title, l2_distance({column}_embedding, %(embedding)s) 
                FROM embeddings.documents 
                ORDER BY l2_distance({column}_embedding, %(embedding)s) LIMIT 10
            """, {"embedding": embedding})
        for row in cur.fetchall():
            print(row)


if __name__ == '__main__':
    # WikiDataGenerator.get_wikipedia_data(max_files=100)
    # WikiDataGenerator.decompress_wiki_data()
    # process_wiki_data()
    print(test_embeddings("swedish band from the 70s", "title"))