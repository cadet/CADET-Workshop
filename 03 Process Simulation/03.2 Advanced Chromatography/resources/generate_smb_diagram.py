"""Generate smb_zones.svg: the four SMB zones of the lesson as a ring, with its inlets and outlets.

The layout follows the lesson's connections: eluent -> zone I -> extract / zone II -> zone III
(+ feed) -> raffinate / zone IV -> zone I. The liquid flows clockwise; at each switch,
CarouselBuilder moves every column one zone against the liquid flow (zone I -> zone IV).

Run from this folder: python generate_smb_diagram.py
"""

from pathlib import Path

FG = "#1f2937"
MUTED = "#5b6370"
BRAND = "#005b82"
WIDTH, HEIGHT = 760, 560
LEFT, TOP, RIGHT, BOTTOM = 200, 110, 560, 430
CX, CY = (LEFT + RIGHT) / 2, (TOP + BOTTOM) / 2

# name, centre, horizontal box, role, caption anchor
ZONES = [
    ("Zone I", (CX, TOP), True, "desorbs B", (CX, TOP - 34)),
    ("Zone II", (RIGHT, CY), False, "desorbs A", (RIGHT + 34, CY)),
    ("Zone III", (CX, BOTTOM), True, "adsorbs B", (CX, BOTTOM + 46)),
    ("Zone IV", (LEFT, CY), False, "adsorbs A", (LEFT - 34, CY)),
]

# Streams at the corners: corner, outer end, inlet?, label lines, text anchor
STREAMS = [
    ((LEFT, TOP), (LEFT - 70, TOP - 70), True, ["Eluent", "fresh buffer"], "end"),
    ((RIGHT, TOP), (RIGHT + 70, TOP - 70), False, ["Extract", "strongly binding B"], "start"),
    ((RIGHT, BOTTOM), (RIGHT + 70, BOTTOM + 70), True, ["Feed", "A + B"], "start"),
    ((LEFT, BOTTOM), (LEFT - 70, BOTTOM + 70), False, ["Raffinate", "weakly binding A"], "end"),
]


def text(x, y, content, size=15, weight="normal", fill=FG, anchor="middle"):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{content}</text>')


def arrow(x1, y1, x2, y2, width=2.5):
    return (f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{FG}" '
            f'stroke-width="{width}" marker-end="url(#arrow)"/>')


def svg():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{FG}"/>'
        '</marker></defs>',
        # The ring of connections
        f'<rect x="{LEFT}" y="{TOP}" width="{RIGHT - LEFT}" height="{BOTTOM - TOP}" fill="none" '
        f'stroke="{FG}" stroke-width="2.5"/>',
    ]
    # Clockwise liquid flow on each side, between corner and zone
    q = 0.22
    for (x1, y1), (x2, y2) in [((LEFT, TOP), (RIGHT, TOP)), ((RIGHT, TOP), (RIGHT, BOTTOM)),
                               ((RIGHT, BOTTOM), (LEFT, BOTTOM)), ((LEFT, BOTTOM), (LEFT, TOP))]:
        for a, b in [(0.06, q), (1 - q, 0.94)]:
            parts.append(arrow(x1 + a * (x2 - x1), y1 + a * (y2 - y1), x1 + b * (x2 - x1), y1 + b * (y2 - y1)))
    for name, (x, y), horizontal, role, (tx, ty) in ZONES:
        w, h = (150, 46) if horizontal else (46, 150)
        parts.append(f'<rect x="{x - w / 2:g}" y="{y - h / 2:g}" width="{w}" height="{h}" rx="6" '
                     f'fill="{BRAND}"/>')
        if horizontal:
            parts.append(text(x, y, name, 17, "bold", "white"))
        else:
            parts.append(f'<g transform="rotate(-90 {x:g} {y:g})">{text(x, y, name, 17, "bold", "white")}</g>')
        anchor = "middle" if horizontal else ("start" if x > CX else "end")
        parts.append(text(tx, ty, role, 14, fill=MUTED, anchor=anchor))
    for (cx, cy), (ox, oy), inlet, (label, detail), anchor in STREAMS:
        parts.append(arrow(ox, oy, cx, cy) if inlet else arrow(cx, cy, ox, oy))
        dx = -8 if anchor == "end" else 8
        dy = -18 if oy < cy else 18
        parts.append(text(ox + dx, oy + dy - 9, label, 16, "bold", anchor=anchor))
        parts.append(text(ox + dx, oy + dy + 9, detail, 13, fill=MUTED, anchor=anchor))
    # Centre: liquid flow and column movement
    parts.append(text(CX, CY - 34, "liquid flows clockwise:", 14, fill=MUTED))
    parts.append(text(CX, CY - 14, "I → II → III → IV", 15, "bold"))
    parts.append(text(CX, CY + 18, "at each switch, every column", 14, fill=MUTED))
    parts.append(text(CX, CY + 38, "moves one zone back: I → IV", 15, "bold"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    Path(__file__).with_name("smb_zones.svg").write_text(svg())
