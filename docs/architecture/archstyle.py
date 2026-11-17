"""Shared style for portfolio architecture diagrams.

Every project diagram imports this so they all look the same: white background, top-down,
user at the top, an "AWS Cloud" boundary, labeled layer clusters and labeled arrows.
Non-AWS services get a plain labeled box, never a borrowed icon.

    from archstyle import arch, aws_cloud, outside, layer, box, e, linkedin

    with arch("FinTech AI", "What runs today", "docs/architecture/architecture"):
        ...
    linkedin("docs/architecture/architecture")
"""
from contextlib import contextmanager
from pathlib import Path

from diagrams import Cluster, Diagram, Edge, Node
from PIL import Image

FONT = "Helvetica"
ATTRIBUTION = "Johnathan Horner, Shoot It Analytics LLC"


@contextmanager
def arch(project: str, subtitle: str, out_base: str):
    """Open a diagram. Writes <out_base>.png at 300 DPI."""
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    label = f"{project}\n{subtitle}\n{ATTRIBUTION}\n "
    graph_attr = {
        "bgcolor": "white",
        "dpi": "300",
        "fontname": FONT,
        "fontsize": "22",
        "labelloc": "t",
        "pad": "0.6",
        "nodesep": "0.7",
        "ranksep": "0.9",
        "splines": "spline",
        "compound": "true",
    }
    node_attr = {"fontname": FONT, "fontsize": "12"}
    edge_attr = {"fontname": FONT, "fontsize": "10", "color": "#3B4252", "fontcolor": "#1F2937",
                 "penwidth": "1.4", "arrowsize": "0.8"}
    with Diagram(label, filename=out_base, outformat="png", show=False, direction="TB",
                 graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr) as d:
        yield d


def aws_cloud(label: str = "AWS Cloud"):
    """The AWS Cloud boundary. Put only AWS resources inside it."""
    return Cluster(label, graph_attr={"bgcolor": "white", "pencolor": "#232F3E", "penwidth": "2",
                                      "style": "solid", "fontname": FONT, "fontsize": "14",
                                      "labeljust": "l", "labelloc": "b", "margin": "18"})


def outside(label: str):
    """A non-AWS hosting boundary, e.g. "Streamlit Community Cloud" or "Google AI (Gemini API)"."""
    return Cluster(label, graph_attr={"bgcolor": "white", "pencolor": "#6B7280", "penwidth": "1.6",
                                      "style": "dashed", "fontname": FONT, "fontsize": "14",
                                      "labeljust": "l", "labelloc": "b", "margin": "18"})


def layer(label: str, margin: int = 30):
    """A labeled layer inside a boundary, e.g. "Agents" or "Retrieval (RAG)".

    The label sits top-left. Arrows enter nodes at their center, so a narrow layer can put an
    arrow through its own label; raise `margin` (points) for that layer until it clears.
    """
    return Cluster(label, graph_attr={"bgcolor": "#F5F7FA", "pencolor": "#AAB4C3", "penwidth": "1",
                                      "style": "rounded", "fontname": FONT, "fontsize": "11",
                                      "labeljust": "l", "labelloc": "t", "margin": str(margin)})


def box(label: str) -> Node:
    """A plain labeled box for a non-AWS component (no icon)."""
    return Node(label, shape="box", style="rounded,filled", fillcolor="white", color="#4B5563",
                penwidth="1.2", fontname=FONT, fontsize="12", margin="0.18,0.10",
                fixedsize="false", width="1.9", height="0.7", labelloc="c")


def e(label: str, **kw) -> Edge:
    """A labeled arrow. Every arrow in a portfolio diagram must use this."""
    return Edge(label=f" {label} ", **kw)


def linkedin(out_base: str, size=(1200, 1500), margin=48) -> str:
    """Fit <out_base>.png onto a white 1200x1500 canvas as <out_base>-linkedin.png."""
    src = Image.open(f"{out_base}.png").convert("RGB")
    w, h = size
    scale = min((w - 2 * margin) / src.width, (h - 2 * margin) / src.height)
    img = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    out = f"{out_base}-linkedin.png"
    canvas.save(out, dpi=(300, 300))
    return out
