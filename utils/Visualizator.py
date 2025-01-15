import re
import os

from pdf2image import convert_from_bytes

from graphviz import Digraph

from operators import Operator, Column, Constant, HardEqual, SoftEqual, ConjunctiveCriteria
from operators.Join import InnerSoftJoin, InnerHashJoin
from operators.Project import Project
from operators.Select import Select
from operators.Dummy import Dummy
from utils.Model import EmbeddingModel
from PIL import Image, ImageChops


def trim_image(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def to_latex_pdf(dot: Digraph, path: str, filename: str):
    remap = { "π": "\\pi", "⋈": "\\bowtie", "σ": "\\sigma", "∧": "\\land", "∨": "\\lor", "≈": "\\approx"}

    dot_description = str(dot)
    for key in remap:
        dot_description = dot_description.replace(key, remap[key])

    dot_description = re.sub(r'label="([^"]+)"', r'label="$\1$"', dot_description)
    dot_description = dot_description.replace("label=", "texlbl=")

    path_tmp = os.path.join(path, "tmp")
    os.makedirs(path_tmp, exist_ok=True)

    with open(f"{path_tmp}/{filename}.dot", 'w') as f:
        f.write(dot_description)

    os.system(f"dot2tex {path_tmp}/{filename}.dot -f tikz --autosize > {path_tmp}/{filename}.tex")
    os.system(f"pdflatex --output-directory={path_tmp} {path_tmp}/{filename}.tex")
    # os.system(f"mv {path_tmp}/{filename}.pdf {path}/")

    with open(f"{path}/{filename}.pdf", 'rb') as f:
        image = convert_from_bytes(f.read(), dpi=600)[0]

    cropped_img = trim_image(image)
    cropped_img.save(os.path.join(path, f"{filename}.png"))

    os.system(f"rm -rf {path_tmp}")


def visualize(operator: Operator, path: str, filename: str):
    structure = operator.get_structure()

    dot = Digraph()
    dot.attr(rankdir='BT')

    parse_nodes(structure, dot, True)

    to_latex_pdf(dot, path, filename)


def parse_nodes(node: tuple[str, list] | str, dot: Digraph, final_node=False) -> str:
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


def visualize_example_hard_query():
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
    visualize(p, '../img/', 'hard_query')


def visualize_example_soft_query():
    em = EmbeddingModel("../config.ini")
    companies = Dummy("Companies", ["name", "country"], [("red bull", "Austria")])
    sanctions = Dummy("Sanctions", ["name", "type", "target"], [("RED BULL GMBH", "Company", True)])
    sel_companies = Select(companies, SoftEqual(Column("country"), Constant("Austria"), em))
    sel_sanctions = Select(sanctions, ConjunctiveCriteria(
        SoftEqual(Column("type"), Constant("Company"), em),
        HardEqual(Column("target"), Constant(True))
    ))

    join = InnerSoftJoin(sel_companies, sel_sanctions, Column("Companies.name"), Column("Sanctions.name"), em)
    p = Project(join, ["Companies.name"])
    visualize(p, '../img/', 'soft_query')


if __name__ == '__main__':
    visualize_example_hard_query()
    visualize_example_soft_query()