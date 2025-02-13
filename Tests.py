import configparser
import copy
import logging
import random
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report

from models import ModelMgr
from db.structure import Column, Constant
from db.criteria import Negation, HardEqual, SoftEqual
from db.operators import Dummy, Scan, Select, Project, Join, InnerHashJoin, InnerSoftJoin
from db.operators.Aggregate import HashAggregate, SumAggregation
from models.text_generation.LLaMA import LLaMATextGenerationModel

from utils import CosineSimilarity, get_config

from models.semantic_validation import GeminiValidationModel, LLaMAValidationModel
from models.embedding import GenericEmbeddingModel, LLaMAEmbeddingModel, SentenceTransformerEmbeddingModel

from db.db import DBConnector


def set_compare(a: list[dict], b: list[dict]):
    a_comparable = Counter(tuple(sorted(d.items())) for d in a)
    b_comparable = Counter(tuple(sorted(d.items())) for d in b)
    return a_comparable == b_comparable


def build_vectorized_result(op):
    arr = []

    while row := op.next_vectorized():
        arr.extend(row)

    return arr

test_data = [{"a": i, "b": i + 1, "c": i + 2} for i in range(0, 111, 3)]

# https://gist.github.com/OdeToCode/582e9c044eee5882d54a6e5997c0be52
test_data_cars = [
    'Volvo', 'Mitsubishi Motors Corporation', 'KIA MOTORS CORPORATION', 'INFINITI', 'Cadillac', 'NISSAN',
    'Acura', 'ALFA ROMEO', 'SCION', 'LEXUS']

# https://gist.github.com/researchranks/ffe24c33df30e64f51271ddec83b4af6
test_data_plants = [
    'wheat', 'violet', 'alder', 'keek', 'box', 'speedwell', 'polkweed', 'eytelia', 'rosemary', 'serviceberry']

test_data_advanced = [(obj, ) for obj in test_data_cars + test_data_plants]
random.shuffle(test_data_advanced)

test_data_companies = [
    # Real Matches
    ("OESTERREICHISCHE AERZTE- UND APOTHEKERBANK A.G.", "österreichische ärzte- und apothekerbank ag"),
    ("Hörburger GmbH & Co KG", "hörburger gmbh"),
    ("Raiffeisenlandesbank Oberösterreich AG", "privat bank ag der raiffeisenlandesbank oberösterreich"),
    ("Raiffeisen-Landesbank Steiermark AG","raiffeisen-landesbank steiermark ag"),
    ("Raiffeisenlandesbank Tiro AG","raiffeisen-landesbank tirol ag"),
    ("Raiffeisen Bank International AG", "raiffeisen bank international ag"),
    ("UniCredit Bank Austria AG", "unicredit bank austria ag"),
    ("STEIERMAERKISCHE BANK UND SPARKASSEN AG","steiermärkische bank und sparkasse"),
    ("Wiener Stadtwerke", "wiener stadtwerke holding ag"),
    ("Wien Energie GmbH", "wien energie gmbh"),
    # Semantic Real Matches
    ("Alphabet Inc.", "Google LLC"),
    ("Meta", "Facebook"), ("Meta", "Instagram"), ("Meta", "WhatsApp"),
    ("Facebook", "WhatsApp"), ("Facebook", "Instagram"),
    ("Toyota", "Lexus"),
    ("PepsiCo", "Lay's"),
    ("The Coca-Cola Company", "Fanta"),
    ("Johnson & Johnson", "Janssen"),
    ("Microsoft", "LinkedIn"), ("Microsoft", "GitHub"), ("Microsoft", "Xbox Game Studios"),
]

test_data_noise = [
    ("Nestlé", "Unilever"),
    ("Visa Inc.", "Mastercard"),
    ("Warner Music Group", "Universal Music Group"),
    ("The Walt Disney Company", "Paramount Pictures"),
    ("TikTok", "Snapchat"),
    ("Nintendo", "Electronic Arts"),
    ("Starbucks Corporation", "Dunkin' Brands"),
    ("McDonald's Corporation", "Burger King"),
    ("IBM", "SAP SE"),
    ("Netflix, Inc.", "HBO"),
    ("Spotify Technology S.A.", "Pandora Media"),
    ("BMW", "Volkswagen"),
    ("Adobe Inc.", "Oracle Corporation"),
    ("Intel Corporation", "Advanced Micro Devices, Inc."),
    ("Nike, Inc.", "Adidas AG"),
    ("Pinterest", "Reddit"),
    ("Bumble Inc.", "Tinder"),
]

template = "Is {a} the same person as {b}?"

artists = [
    ("Robyn Fenty", "Rihanna"),
    ("Marshall Mather", "Eminem"),
    ("Marshall Bruce Mathers III", "Eminem"),
    ("Stefani Germanotta", "Lady Gaga"),
    ("Stefani Joanne Angelina Germanotta", "Lady Gaga"),
    ("Stefano Germanotta", None),  # Trick the LLM with Lady Gaga
    ("Adele Laurie Blue Adkins MBE", "Adele"),
    ("Katheryn Elizabeth Hudson", "Katy Perry"),
    ("Taylor Swift", None),
    ("Onika Maraj", "Nicki Minaj"),
    ("Volkan Yaman", "Apache 207"),
    ("Johann Hölzel", "Falco"),
    ("Franz Bibiza", "BIBIZA"),
    ("Shawn Corey Carter", "Jay-Z"),
    ("Aubrey Drake Graham", "Drake"),
    ("Belcalis Marlenis Almánzar", "Cardi B"),
    ("Calvin Cordozar Broadus Jr.", "Snoop Dogg"),
    ("Jacques Berman Webster II", "Travis Scott"),
    ("Ella Marija Lani Yelich-O'Connor", "Lorde"),
    ("Peter Gene Hernandez", "Bruno Mars"),
    ("Melissa Viviane Jefferson", "Lizzo"),
    ("Abel Makkonen Tesfaye", "The Weeknd"),
    ("Alecia Beth Moore", "Pink"),
    ("P!nk", "Pink"),
    ("Austin Richard Post", "Post Malone"),
    ("Claire Elise Boucher", "Grimes"),
    ("Paul David Hewson", "Bono"),
    ("Reginald Kenneth Dwight", "Elton John"),
]

statements = []
for name, alias in artists:
    if alias is not None:
        statements.append((template.format(a=name, b=alias), True))

for name1, alias1 in artists:
    for name2, alias2 in artists:
        if name1 == name2:
            continue

        statements.append((template.format(a=name1, b=name2), False))
        if alias2 is not None:
            statements.append((template.format(a=name1, b=alias2), False))

random.shuffle(statements)


def test_dummy():
    print("Test Dummy...", end=" ")
    dummy = Dummy("test", ["a", "b", "c"], test_data)

    compare_data = [row for row in dummy]
    assert compare_data == test_data

    dummy.open()
    assert test_data == build_vectorized_result(dummy)
    print("Done")

def test_scan(db_connector, embedding_model, semantiv_validation):
    print("Test Scan...", end=" ")
    scan1 = Scan("firms", db_connector, embedding_model, semantiv_validation, threshold=.3)
    assert scan1.table.table_name == "companies"
    scan1.close()

    scan2 = Scan("actors", db_connector, embedding_model, semantiv_validation, threshold=.3)
    assert scan2.table.table_schema == "imdb", scan2.table.table_name == "persons"
    scan2.close()

    with db_connector.get_cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS public.tmp_table ( a INTEGER, b INTEGER, c INTEGER );")
        cursor.execute("DELETE FROM public.tmp_table")
        for row in test_data:
            cursor.execute("INSERT INTO tmp_table (a, b, c) VALUES (%(a)s, %(b)s, %(c)s);", row)
        db_connector.conn.cursor()

    scan3 = Scan("public.tmp_table", db_connector, embedding_model, semantiv_validation, use_semantic_table_search=False)
    assert scan3.table.table_name == "tmp_table"
    compare_data = [dict(row) for row in scan3]
    assert set_compare(compare_data, test_data)

    scan3.open()
    assert set_compare(test_data, build_vectorized_result(scan3))
    print("Done")

def test_select(embedding_model):
    print("Test Select...", end=" ")
    dummy = Dummy("test", ["a", "b", "c"], test_data)

    sel_equal = Select(dummy, HardEqual(Column("a"), Constant(3)))
    x = [rec for rec in sel_equal]
    assert [{'a': 3, 'b': 4, 'c': 5}] == x

    sel_equal.open()
    assert [{'a': 3, 'b': 4, 'c': 5}] == [rec for rec in build_vectorized_result(sel_equal)]

    dd_neu = copy.deepcopy(test_data)
    dd_neu.remove({'a': 3, 'b': 4, 'c': 5})
    sel_unequal = Select(dummy, Negation(HardEqual(Column("a"), Constant(3))))
    assert dd_neu, [rec for rec in sel_unequal]

    sel_unequal.open()
    assert dd_neu, [rec for rec in build_vectorized_result(sel_unequal)]

    dummy_advanced = Dummy("test", ["object"], test_data_advanced)

    crit = SoftEqual(Column("object"), Constant("car brand"), embedding_model, CosineSimilarity(),.3)
    sel_advanced = Select(dummy_advanced, crit)
    result = {x["object"] for x in sel_advanced}
    assert set(test_data_cars).issubset(result) and len(result) < len(test_data_advanced)

    sel_advanced.open()
    result = {x["object"] for x in build_vectorized_result(sel_advanced)}
    # TODO: Add Sematic Validation -> Test for full equality
    assert set(test_data_cars).issubset(set(result)) and len(result) < len(test_data_advanced)

    # TODO: Add Sematic Validation -> Test for full equality
    crit = Negation(
        SoftEqual(Column("object"), Constant("car brand"), embedding_model, CosineSimilarity(), 0.4))
    result = {x["object"] for x in Select(dummy_advanced, crit)}
    assert set(test_data_plants).issubset(set(result)) and len(result) < len(test_data_advanced)

    print("Done")

def test_projection(em):
    print("Test Projection...", end=" ")
    reduced_test_data = list(map(lambda x: {"a": x["a"], "b": x["b"]}, copy.deepcopy(test_data)))
    project = Project(Dummy("test", ["a", "b", "c"], test_data), ["a", "b"], em)
    assert reduced_test_data == [x for x in project]
    project.open()
    assert reduced_test_data == build_vectorized_result(project)

    dummy = Dummy("test", ["year_founded", "locality", "industry"], [(2000, 'austria', 'industry')])
    project = Project(dummy, ["founded", "where"], em)
    assert [col.column_name for col in project.table.table_structure] == ["year_founded", "locality"]
    print("Done")



def test_join(embedding_model):
    print("Test Join...", end=" ")
    # Test Normal Join
    d1 = Dummy("d1", ["a", "b", "c"], [(i,i+1,i+2) for i in range(0, 111, 3)])
    d2 = Dummy("d2", ["d", "e", "f"], [(i,i+1,i+2) for i in range(0, 111, 3)])
    join = Join(d1, d2, HardEqual(Column("a"), Column("d")))
    assert [{"a": i, "d": i, "b": i+1, "e": i+1, "c": i+2, "f": i+2} for i in range(0, 111, 3)] == [rec for rec in join]

    d1 = Dummy("d1", ["x", "y"], [(i, i + 1) for i in range(0, 100, 2)])
    d2 = Dummy("d2", ["z", "y"], [(i, i + 1) for i in range(0, 100, 2)])
    join = Join(d1, d2, HardEqual(Column("x"), Column("z")))
    assert [{"x": i, "d1.y": i + 1, "z": i, "d2.y": i + 1} for i in range(0, 100, 2)] == [rec for rec in join]

    d1 = Dummy("d1", ["x", "y"], [(i, i + 1) for i in range(0, 100, 2)])
    d2 = Dummy("d2", ["x", "z"], [(i, i + 1) for i in range(0, 100, 2)])
    join = Join(d1, d2, HardEqual(Column("d1.x"), Column("d2.x")))
    assert [{"d1.x": i, "d2.x": i, "y": i + 1, "z": i + 1} for i in range(0, 100, 2)] == [rec for rec in join]

    # Test Specialized Hash Join
    d1 = Dummy("d1", ["a", "b"], [(i, i + 1) for i in range(0, 100, 2)])
    d2 = Dummy("d2", ["c", "d"], [(i, i + 1) for i in range(0, 100, 2)])
    join = InnerHashJoin(d1, d2, Column("a"), Column("c"))
    assert [{"a": i, "c": i, "b": i + 1, "d": i + 1} for i in range(0, 100, 2)] == [rec for rec in join]

    d1 = Dummy("d1", ["x", "y"], [(i, i + 1) for i in range(0, 100, 2)])
    d2 = Dummy("d2", ["x", "z"], [(i, i + 1) for i in range(0, 100, 2)])
    join = InnerHashJoin(d1, d2, Column("d1.x"), Column("d2.x"))
    assert [{"d1.x": i, "d2.x": i, "y": i + 1, "z": i + 1} for i in range(0, 100, 2)] == [rec for rec in join]

    # Test Specialized Soft Hash Join
    c1 = Dummy("c1", ["companies"], [(c[0], ) for c in set(test_data_companies + test_data_noise)])
    c2 = Dummy("c2", ["companies"], [(c[1], ) for c in set(test_data_companies + test_data_noise)])
    join = InnerSoftJoin(c1, c2, Column("c1.companies"), Column("c2.companies"), embedding_model, threshold=50, debug=True)
    # TODO: Validate [rec for rec in join]
    print("Done")

def test_aggregations():
    d = Dummy("Test", ["a", "b", "c"], [(1,2,3), (1,3,4), (1,5,6), (2,7,8), (2,8,9), (3,0,0)])
    ha = HashAggregate(d, ["a"], [SumAggregation("b"), SumAggregation("c")])
    assert [x for x in ha] == [{'a': 1, 'b': 10, 'c': 13}, {'a': 2, 'b': 15, 'c': 17}, {'a': 3, 'b': 0, 'c': 0}]


def test_semantic_validation(validator):
    print("Evaluation on class [cars/ plants] membership")
    test_data = [(x , "car") for x in test_data_cars] + [(x, "plant") for x in test_data_plants]
    df_test_data = pd.DataFrame(test_data, columns=["object", "type"]).sample(frac=1)
    df_test_data["is_car"] = df_test_data["object"].apply(lambda x: validator(f"Is {x} a car?"))
    df_test_data["is_plant"] = df_test_data["object"].apply(lambda x: validator(f"Is {x} a plant?"))

    df_test_data["prediction"] = df_test_data.apply(
        lambda x: "car" if x["is_car"] and not x["is_plant"] else ("plant" if not x["is_car"] and x["is_plant"] else "-"),
        axis=1)

    print("Quantitative Evaluation")
    print(classification_report(df_test_data["type"], df_test_data["prediction"]))

    print("Qualitative Evaluation")
    print(df_test_data[df_test_data["prediction"] == "-"])

    print("\nEvaluation on entity matching")
    results = []
    for statement, answer in tqdm(statements):
        results.append((statement, answer, validator(statement)))

    df = pd.DataFrame(results, columns=["statement", "answer", "prediction"])
    print("Quantitative Evaluation")
    print(classification_report(df["answer"], df["prediction"]))

    print("Qualitative Evaluation")
    print(df[df["prediction"] != df["answer"]])

def test_models(model_mgr: ModelMgr, config: configparser.ConfigParser):
    # Validators
    prompt = "Is Lady Gaga the same person as Stefani Joanne Angelina Germanotta"
    llama_validation = LLaMAValidationModel(model_mgr)
    gemini_validation = GeminiValidationModel(config["MODEL"]["google_aistudio_api_key"])

    # Embeddings
    text = "Lady Gaga"
    llama_embedding = LLaMAEmbeddingModel(model_mgr)
    generic_embedding = GenericEmbeddingModel(model_mgr)
    st_embedding = SentenceTransformerEmbeddingModel(model_mgr)

    print("LLaMAValidationModel", llama_validation(prompt))
    print("GeminiValidationModel", gemini_validation(prompt))

    print("LLaMAEmbeddingModel", llama_embedding(text).shape)
    print("GenericEmbeddingModel", generic_embedding(text).shape)
    print("SentenceTransformerEmbeddingModel", st_embedding(text).shape)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    config_file = "./config.ini"
    config: configparser.ConfigParser = get_config(config_file)

    m = ModelMgr()
    db = DBConnector(config_file)
    em = SentenceTransformerEmbeddingModel(m)
    sv = LLaMAValidationModel(m)
    gm = LLaMATextGenerationModel(m)

    test_dummy()
    test_scan(db, em, sv)
    test_projection(em)
    test_select(em)
    test_join(em)

    # test_semantic_validation(sv)

    # gv = Gemini_Validator("./config.ini", model_name="gemini-2.0-flash-exp")
    # test_semantic_validation(gv)
