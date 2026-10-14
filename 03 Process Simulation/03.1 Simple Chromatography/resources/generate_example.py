"""Regenerate the lesson's opening chromatogram from its completed model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CADETProcess.simulator import Cadet
from chromatography_setup import separation_model, plot_lwe
from cadetgui.widgets.diagrams.flow_sheet import render_flow_sheet_svg

column, process = separation_model()
diagram = render_flow_sheet_svg(process.flow_sheet)
# The renderer returns an HTML wrapper; an image file needs only its SVG element.
diagram = diagram[diagram.index("<svg"):diagram.rindex("</svg>") + len("</svg>")]
Path(__file__).with_name("separation_system.svg").write_text(diagram)
process.cycle_time = 52 * 60
process.add_event("load", "flow_sheet.inlet.c", [50, 0.1, 0.1], 0)
process.add_event("wash", "flow_sheet.inlet.c", [50, 0, 0], 2 * 60)
process.add_event("elute", "flow_sheet.inlet.c", [[50, (500 - 50) / (30 * 60)], [0], [0]], 12 * 60)
process.add_event("final_wash", "flow_sheet.inlet.c", [500, 0, 0], 42 * 60)
results = Cadet().simulate(process)
fig, _ = plot_lwe(results)
fig.savefig(Path(__file__).with_name("lwe_separation.png"), dpi=160)
plt.close(fig)
