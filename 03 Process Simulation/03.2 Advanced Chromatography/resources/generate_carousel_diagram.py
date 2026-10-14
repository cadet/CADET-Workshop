"""Render the exercise's configured zones without running a simulation."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import jupytext
from cadetgui.widgets.diagrams.flow_sheet import render_flow_sheet_svg

resources = Path(__file__).resolve().parent
scope = {}
exec((resources / "advanced_chromatography.py").read_text(), scope)
notebook = jupytext.read(resources.parent / "03.2b_advanced_chromatography_exercises.md")
start = next(i for i, cell in enumerate(notebook.cells)
             if cell.cell_type == "markdown" and cell.source.startswith("## Example 2:"))
for cell in notebook.cells[start:]:
    if cell.cell_type != "code":
        continue
    if "carousel_results =" in cell.source or "diagram =" in cell.source:
        break
    exec(cell.source, scope)

view = copy.deepcopy(scope["builder"].flow_sheet)
labels = {
    "i_wash": "Wash buffer", "i_feed": "Feed", "i_elute": "Elution buffer",
    "z_wash": "Wash: 3 series", "z_feed": "Feed: 3 parallel",
    "z_dilute": "Dilute: 2 reverse", "z_elute": "Elute: 2 series",
    "o_waste": "Waste (99%)", "o_product": "Product",
}
for unit in view.units:
    unit.name = labels[unit.name]
svg = render_flow_sheet_svg(view)
svg = svg[svg.index("<svg"):svg.rindex("</svg>") + len("</svg>")]
namespace = "http://www.w3.org/2000/svg"
ET.register_namespace("", namespace)
root = ET.fromstring(svg)
# Label the four generic zone boxes using the same diagram symbols and styling.
zone_boxes = root.findall(f"{{{namespace}}}rect")
assert len(zone_boxes) == 4
for box, label in zip(zone_boxes, ["Wash", "Feed", "Dilute", "Elute"]):
    text = ET.SubElement(root, f"{{{namespace}}}text", {
        "x": f"{float(box.get('x')) + float(box.get('width')) / 2:g}",
        "y": f"{float(box.get('y')) + float(box.get('height')) / 2:g}",
        "text-anchor": "middle", "dominant-baseline": "central",
        "font-size": "18", "font-weight": "bold", "fill": "#1f2937",
    })
    text.text = label
x, y, width, height = map(float, root.attrib["viewBox"].split())
root.set("viewBox", f"{x:g} {y:g} {width:g} {height + 48:g}")
line1 = ET.SubElement(root, f"{{{namespace}}}text", {
    "x": "16", "y": f"{height + 12:g}", "font-size": "14", "fill": "#1f2937",
})
line1.text = "Feed/Wash: 50 mM salt; Elution buffer: 500 mM salt. All inlets: 1.5 mL/min."
line2 = ET.SubElement(root, f"{{{namespace}}}text", {
    "x": "16", "y": f"{height + 32:g}", "font-size": "14", "fill": "#1f2937",
})
line2.text = "Dilute outlet: 99% to Waste, 1% to Elute. Target is carried with the resin."
(resources / "bind_elute_carousel.svg").write_text(ET.tostring(root, encoding="unicode"))
