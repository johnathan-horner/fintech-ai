"""Shared look for every portfolio architecture diagram.

Johnathan Horner, Shoot It Analytics LLC.
Python diagrams library, official AWS icons, white background, top-down, labeled arrows.
Brand icons: Simple Icons (CC0). Outline icons: Lucide (ISC).
"""
from pathlib import Path

from diagrams import Cluster, Edge
from diagrams.custom import Custom

ICONS = Path(__file__).parent / "icons"
FONT = "Inter"

INK = "#232F3E"      # AWS squid ink
ORANGE = "#ED7100"   # primary AI path
GREY = "#7D8998"
RED = "#DD344C"

GRAPH = {
    "dpi": "300", "bgcolor": "white", "fontname": FONT + " SemiBold", "fontsize": "30",
    "fontcolor": INK, "labelloc": "t", "pad": "0.6", "nodesep": "1.25", "ranksep": "1.05",
    "splines": "spline", "newrank": "true", "compound": "true",
}
NODE = {"fontname": FONT, "fontsize": "14", "fontcolor": INK, "width": "1.15", "height": "1.15",
        "imagescale": "true", "labelloc": "b"}
EDGE = {"fontname": FONT, "fontsize": "13", "fontcolor": "#3B4754", "color": GREY, "penwidth": "1.6",
        "arrowsize": "0.9"}


def area(label, kind="layer"):
    """Cluster styles: 'host' (outer platform), 'aws' (AWS Cloud boundary), 'ext' (outside vendor), 'layer'."""
    base = {"fontname": FONT + " SemiBold", "fontsize": "16", "fontcolor": INK, "margin": "48",
            "labeljust": "l", "style": "rounded"}
    styles = {
        "host": {"bgcolor": "#F4F6F9", "pencolor": "#AEB6C1", "style": "rounded,dashed", "penwidth": "1.5"},
        "aws": {"bgcolor": "#FFFFFF", "pencolor": INK, "penwidth": "2.2", "style": "solid"},
        "ext": {"bgcolor": "#FAF8FD", "pencolor": "#B9A9D6", "style": "rounded,dashed", "penwidth": "1.5"},
        "layer": {"bgcolor": "#FFFFFF", "pencolor": "#C9D0D9", "penwidth": "1.2"},
        "plan": {"bgcolor": "#FFFFFF", "pencolor": GREY, "style": "rounded,dashed", "penwidth": "1.5"},
    }
    return Cluster(label, graph_attr={**base, **styles[kind]})


def icon(name, label):
    return Custom(label, str(ICONS / f"{name}.png"))


def flow(label="", **kw):
    return Edge(label=f" {label} " if label else "", **kw)


def primary(label=""):
    return flow(label, color=ORANGE, penwidth="2.6", fontcolor="#B35400")


def soft(label=""):
    return flow(label, style="dashed")
