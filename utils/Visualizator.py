from typing import List

from graphviz import Digraph

from operators import Operator, Column, Constant
from operators.Dummy import Dummy
from operators.Join import InnerHashJoin, SoftInnerJoin
from operators.Project import Project
from operators.Select import Select, HardEqual
from operators.Transform import Transform


def visualize(operator: Operator, filename: str):
    structure = operator.get_structure()

    dot = Digraph(format='png')
    parse_nodes(structure, dot)
    dot.render(filename, cleanup=True)


def parse_nodes(node: tuple[str, List] | str, dot: Digraph) -> str:
    if isinstance(node, tuple):
        key, child_nodes = node
        node_id, node_description = key.split(":")
        dot.node(node_id, node_description)
        for child_node in child_nodes:
            dot.edge(node_id, parse_nodes(child_node, dot))
        return node_id
    else:
        node_id, node_description = node.split(":")
        dot.node(node_id, node_description)
        return node_id

if __name__ == '__main__':
    def x(r: dict) -> dict:
        r["a"] = 2
        return r

    exec_plan = Project(
        SoftInnerJoin(
            Transform(Dummy("d1", ["a", "b", "c"], [(1, 2, 3), (2, 3, 4)]), x),
            Select(
                Dummy("d2", ["a", "d", "e"], [(1, 2, 3), (2, 3, 4)]),
                HardEqual(Column("a"), Constant(2))), Column("a"), Column("a"),
            None, None),
        ["d1.a", "d1.b"])

    visualize(exec_plan, '../img/exec_plan')
