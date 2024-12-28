from typing import List

from operators import Operator, Column, Constant
from operators.Dummy import Dummy
from operators.Project import Project
from operators.Select import HardEqual, Select, HardUnEqual

data_tuples = [(i, i + 1, i + 2) for i in range(0, 98, 3)]
data_dicts = [{"a": i, "b": i + 1, "c": i + 2} for i in range(0, 98, 3)]


def asser_it(op: Operator, gt: List[dict]):
    set_op = {frozenset(d.items()) for d in op}
    set_gt = {frozenset(d.items()) for d in gt}
    assert set_op == set_gt


def assert_vect(op: Operator, gt: List[dict]):
    data_op = []
    data_slice = op.next()
    while data_slice is not None:
        data_op.extend(data_slice)
        data_slice = op.next()

    assert {frozenset(d.items()) for d in data_op} == {frozenset(d.items()) for d in gt}

def test_dummy():
    dummy = Dummy("dummy", ["a", "b", "c"], data_tuples)
    asser_it(dummy, data_dicts)

    dummy = Dummy("dummy", ["a", "b", "c"], data_dicts)
    asser_it(dummy, data_dicts)

    dummy = Dummy("dummy", ["a", "b", "c"], data_dicts, num_tuples=10)
    assert_vect(dummy, data_dicts)

    dummy = Dummy("dummy", ["a", "b", "c"], data_dicts, num_tuples=30)
    assert_vect(dummy, data_dicts)

    dummy = Dummy("dummy", ["a", "b", "c"], data_dicts, num_tuples=len(data_dicts))
    assert_vect(dummy, data_dicts)

    dummy = Dummy("dummy", ["a", "b", "c"], data_dicts, num_tuples=len(data_dicts)+10)
    assert_vect(dummy, data_dicts)

    print("All tests for Operator 'Dummy' passed")

def test_project():
    project = Project(Dummy("dummy", ["a", "b", "c"], data_tuples), ["a", "c"])
    asser_it(project, list(map(lambda x: {"a": x["a"], "c": x["c"]}, data_dicts)))

    project = Project(Dummy("dummy", ["a", "b", "c"], data_tuples), ["a", "c"])
    assert_vect(project, list(map(lambda x: {"a": x["a"], "c": x["c"]}, data_dicts)))

    project = Project(Dummy("dummy", ["a", "b", "c"], data_tuples, num_tuples=15), ["a", "c"])
    assert_vect(project, list(map(lambda x: {"a": x["a"], "c": x["c"]}, data_dicts)))

    print("All tests for Operator 'Project' passed")


def test_select():
    select = Select(Dummy("dummy", ["a", "b", "c"], data_tuples), HardEqual(Column("a"), Constant(63)))
    asser_it(select, [{'a': 63, 'b': 64, 'c': 65}])

    select = Select(Dummy("dummy", ["a", "b", "c"], data_tuples), HardEqual(Column("a"), Constant(63)))
    assert_vect(select, [{'a': 63, 'b': 64, 'c': 65}])

    select = Select(Dummy("dummy", ["a", "b", "c"], data_tuples), HardEqual(Column("a"), Constant(62)))
    asser_it(select, [])

    gt_2 = data_dicts.copy()
    gt_2.remove({"a": 63, "b": 64, "c": 65})

    select = Select(Dummy("dummy", ["a", "b", "c"], data_tuples), HardUnEqual(Column("b"), Constant(64)))
    assert_vect(select, gt_2)

    print("All tests for Operator 'Select' passed")

if __name__ == '__main__':
    test_dummy()
    test_project()
    test_select()