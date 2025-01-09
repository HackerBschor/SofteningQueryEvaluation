import copy
import random

from operators import Negation, HardEqual, Column, Constant, SoftEqual
from operators.Dummy import Dummy
from operators.Scan import Scan
from operators.Transform import Transform
from operators.Select import Select
from operators.Project import Project
from operators.Join import Join, InnerHashJoin, SoftInnerJoin

from collections import Counter

from utils import CosineSimilarity, EuclidianDistance
from utils.Model import EmbeddingModel


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




def test_dummy():
    dummy = Dummy("test", ["a", "b", "c"], test_data)

    compare_data = [row for row in dummy]
    assert compare_data == test_data

    dummy.open()
    assert test_data == build_vectorized_result(dummy)

def test_scan(db_connector, embedding_model):
    scan1 = Scan("firms", db_connector, embedding_model, distance=CosineSimilarity(), threshold=0.8)
    assert scan1.table_name == "companies"
    scan1.close()

    scan2 = Scan("actors", db_connector, embedding_model, distance=EuclidianDistance(), threshold=80)
    assert scan2.schema_name == "imdb", scan2.table_name == "persons"
    scan2.close()

    with db_connector.get_cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE tmp_table ( a INTEGER, b INTEGER, c INTEGER );")
        for row in test_data:
            cursor.execute("INSERT INTO tmp_table (a, b, c) VALUES (%(a)s, %(b)s, %(c)s);", row)

    scan3 = Scan("tmp_table", db_connector, embedding_model)
    assert scan3.table_name == "tmp_table"
    compare_data = [dict(row) for row in scan3]
    assert set_compare(compare_data, test_data)

    scan3.open()
    assert set_compare(test_data, build_vectorized_result(scan3))

def test_transform():
    def dummy_function(row):
        row['c'] = row['a'] + row['b']
        return row

    mapped_test_data = copy.deepcopy(test_data)
    mapped_test_data = list(map(dummy_function, mapped_test_data))

    dummy = Transform(Dummy("test", ["a", "b", "c"], test_data), dummy_function)

    compare_data = [row for row in dummy]
    assert compare_data == mapped_test_data

    dummy.open()
    assert compare_data == build_vectorized_result(dummy)

def test_select(embedding_model):
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

    crit = SoftEqual(Column("object"), Constant("car manufacturers"), embedding_model, CosineSimilarity(), 0.75)
    sel_advanced = Select(dummy_advanced, crit)
    assert set(test_data_cars).issubset(set([x["object"] for x in sel_advanced]))

    sel_advanced.open()
    result = [x["object"] for x in build_vectorized_result(sel_advanced)]
    # TODO: Add Sematic Validation -> Test for full equality
    assert set(test_data_cars).issubset(set(result)) and len(result) < len(test_data_advanced)

    # TODO: Add Sematic Validation -> Test for full equality
    crit = Negation(
        SoftEqual(Column("object"), Constant("car manufacturers"), embedding_model, CosineSimilarity(), 0.8))
    sel_advanced = Select(dummy_advanced, crit)
    result = [x["object"] for x in sel_advanced]
    assert set(test_data_plants).issubset(set(result)) and len(result) < len(test_data_advanced)

def test_projection():
    dummy = Project(Dummy("test", ["a", "b", "c"], test_data), Project.WILDCARD)
    assert test_data == [x for x in dummy]

    reduced_test_data = list(map(lambda x: {"a": x["a"], "b": x["b"]}, copy.deepcopy(test_data)))
    dummy = Project(Dummy("test", ["a", "b", "c"], test_data), ["a", "b"])
    assert reduced_test_data == [x for x in dummy]

    dummy.open()
    assert reduced_test_data == build_vectorized_result(dummy)

def test_join(embedding_model):
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
    join = SoftInnerJoin(c1, c2, Column("c1.companies"), Column("c2.companies"), embedding_model, threshold=50, debug=True)
    # TODO: Validate [rec for rec in join]



if __name__ == '__main__':
    #db = DBConnector("./config.ini")
    em = EmbeddingModel("./config.ini")
    #test_dummy()
    #test_scan(db, em)
    #test_transform()
    #test_select(em)
    # test_projection()
    test_join(em)
