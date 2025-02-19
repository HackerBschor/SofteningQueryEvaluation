import os
import logging

import requests
import zipfile

import pandas as pd

from db.operators import Dummy
from db.operators import InnerSoftJoin

from models import ModelMgr
from models.embedding.SentenceTransformer import SentenceTransformerEmbeddingModel
from models.semantic_validation import DeepSeekValidationModel

# logging.basicConfig(level=logging.INFO)

test_cases = ['Abt-Buy', 'Amazon-GoogleProducts', 'DBLP-ACM', 'DBLP-Scholar']
test_case_files = {
    "Abt-Buy": ["Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv"]
}

def download_test_case(test_case: test_cases, path = "../data/semantic_join"):
    os.makedirs(path, exist_ok=True)

    test_case_file = f"{path}/{test_case}.zip"
    test_case_data_path = f"{path}/{test_case}_data"

    if not os.path.exists(test_case_data_path):
        response = requests.get(f"https://dbs.uni-leipzig.de/files/datasets/{test_case}.zip", stream=True)
        with open(test_case_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        os.makedirs(test_case_data_path, exist_ok=True)
        with zipfile.ZipFile(test_case_file, "r") as zip_ref:
            zip_ref.extractall(test_case_data_path)

    return test_case_data_path

if __name__ == '__main__':
    test_case = test_cases[0]
    data_path = download_test_case(test_case)
    data_files = test_case_files[test_case]

    df_1 = pd.read_csv(f"{data_path}/{data_files[0]}", encoding='unicode_escape')
    df_2 = pd.read_csv(f"{data_path}/{data_files[1]}", encoding='unicode_escape')
    gt = pd.read_csv(f"{data_path}/{data_files[2]}", encoding='unicode_escape')
    gt = {(x[0], x[1]) for x in gt.itertuples(index=False, name=None)}

    m = ModelMgr()
    stem = SentenceTransformerEmbeddingModel(m)
    sv = DeepSeekValidationModel(m)

    abt = Dummy("abt", list(df_1.columns), [row for row in df_1.itertuples(index=False, name=None)])
    buy = Dummy("buy", list(df_2.columns), [row for row in df_2.itertuples(index=False, name=None)])


    isj = InnerSoftJoin(abt, buy, method="both", embedding_method="FULL_SERIALIZED", em=stem, threshold=0.9, columns_left=["name", "description"], columns_right=["name", "description"], sv=sv)
    isj.open()

    result = set()
    for x in isj:
        print(x)
        result.add((x["abt.id"], x["buy.id"]))

    tps, fns, fps = gt & result, gt - result, result - gt
    tp, fn, fp = len(tps), len(fns), len(fps)

    values = {"tp": tp, "fn": fn, "fp": fp}
    results = {"tps": tps, "fns": fns, "fps": fps}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    scores = { "Precision": precision, "Recall": recall, "F1 Score": f1_score, }
    print(scores)