from typing import List

from graphviz import Digraph

from operators import Operator, Column, Constant
from operators.Join import SoftInnerJoin
from operators.Project import Project
from operators.Scan import Scan
from operators.Select import Select, HardEqual, DisjunctiveCriteria, SoftEqualEmbedding
from utils.DB import DBConnector
from utils.Model import GenerationModel, EmbeddingModel


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
    db = DBConnector("../config.ini", use_vector=True)
    em = EmbeddingModel("../config.ini")
    gm = GenerationModel("../config.ini")
    i = Scan("institutions", db, embedding_model=em, generation_model=gm)
    r = Scan("reports", db, embedding_model=em, generation_model=gm)
    sel_i = Select(i, HardEqual(Column("type"), Constant("finance")))
    sel_r = Select(r, DisjunctiveCriteria(
        SoftEqualEmbedding(Column("text"), Constant("is about Google"), em),
        SoftEqualEmbedding(Column("text"), Constant("is about Amazon"), em)))
    join = SoftInnerJoin(sel_i, sel_r, Column("institutions.name"), Column("reports.issued_by"), em, gm)
    p = Project(join, ["institutions.name"])

    visualize(p, '../img/exec_plan')
