from typing import List

from graphviz import Digraph

from operators import Operator, Column, Constant, HardEqual, SoftEqual, DisjunctiveCriteria, ConjunctiveCriteria
from operators.Join import SoftInnerJoin, InnerHashJoin
from operators.Project import Project
from operators.Scan import Scan
from operators.Select import Select
from operators.Dummy import Dummy
from utils.DB import DBConnector
from utils.Model import GenerationModel, EmbeddingModel


def visualize(operator: Operator, filename: str, file_format='png'):
    structure = operator.get_structure()

    dot = Digraph(format=file_format)
    dot.attr(rankdir='BT')
    # dot.attr(dpi='300')

    parse_nodes(structure, dot, True)

    print(dot)
    #
    dot.render(filename, cleanup=True)


def parse_nodes(node: tuple[str, List] | str, dot: Digraph, final_node=False) -> str:
    if isinstance(node, tuple):
        key, child_nodes = node
        node_id, node_description = key.split(":")
        dot.node(node_id, node_description, color='red' if final_node else None)
        for child_node in child_nodes:
            dot.edge(parse_nodes(child_node, dot), node_id)
        return node_id
    else:
        node_id, node_description = node.split(":")
        dot.node(node_id, node_description, color='blue')
        return node_id


def visualize_example_1():
    # em = EmbeddingModel("../config.ini")
    companies = Dummy("Companies", ["name", "country"], [("red bull", "Austria")])
    sanctions = Dummy("Sanctions", ["name", "type", "target"], [("RED BULL GMBH", "Company", True)])
    sel_companies = Select(companies, HardEqual(Column("country"), Constant("Austria")))
    sel_sanctions = Select(sanctions, ConjunctiveCriteria(
        HardEqual(Column("type"), Constant("Company")),
        HardEqual(Column("target"), Constant(True))
    ))

    join = InnerHashJoin(sel_companies, sel_sanctions, Column("Companies.name"), Column("Sanctions.name"))
    p = Project(join, ["Companies.name"])
    visualize(p)


if __name__ == '__main__':
    visualize_example_1()