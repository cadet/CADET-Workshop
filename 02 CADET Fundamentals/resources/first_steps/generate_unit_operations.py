"""Generate unit_operation_examples.svg from the CADET-GUI diagram symbols.

Run with the workshop environment: ``python generate_unit_operations.py``.
"""

import re
from pathlib import Path

from cadetgui.widgets.diagrams.svg import PORTS, load_symbol, place_symbol, text_block, tube_segment

PORT_Y = 70  # Common height of the inlet/outlet ports.
GAP = 45
MARGIN = 50  # Room for captions wider than the outer symbols.
TUBE_LENGTH = 110
EXAMPLES = [
    # (symbol, title, examples)
    ("Inlet", "Inlet", ["pump, feed,", "buffer reservoir"]),
    ("Mixer", "Cstr", ["mixer, tank,", "batch reactor"]),
    (None, "TubularReactor", ["tubing,", "pipes"]),
    ("Column", "Column models", ["LRM, LRMP, GRM:", "packed columns"]),
    ("Outlet", "Outlet", ["detector,", "fraction collector"]),
]


def build_svg():
    """Place the symbols on one port line, each with a caption below."""
    parts, captions = [], []
    x, bottom = MARGIN, 0.0
    for symbol, title, lines in EXAMPLES:
        if symbol is None:
            width = TUBE_LENGTH
            parts.append(tube_segment(x, x + width, PORT_Y))
            bottom = max(bottom, PORT_Y)
        else:
            sym = load_symbol(symbol)
            width = sym.width
            y = PORT_Y - PORTS[symbol][2]
            parts.append(place_symbol(symbol, x, y).replace(">Mixer<", ">CSTR<"))
            bottom = max(bottom, y + sym.height)
        captions.append((x + width / 2, title, lines))
        x += width + GAP
    caption_y = bottom + 28
    for cx, title, lines in captions:
        captions_svg = text_block([title], cx, caption_y, strong=True)
        captions_svg += text_block(lines, cx, caption_y + 18)
        parts.append(captions_svg)
    width, height = x - GAP + MARGIN, caption_y + 50
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="Examples of unit operations: inlet, CSTR, tubing, column and outlet" '
        f'font-family="system-ui, sans-serif"><rect width="100%" height="100%" fill="white"/>'
        + "".join(parts) + "</svg>"
    )
    # Replace theme variables by their fallback colours for use as a static image.
    return re.sub(r"var\(--[\w-]+,\s*(#[0-9a-fA-F]+)\)", r"\1", svg)


if __name__ == "__main__":
    target = Path(__file__).with_name("unit_operation_examples.svg")
    target.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {target}")
