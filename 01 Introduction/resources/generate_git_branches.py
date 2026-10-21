"""Generate git_branches.svg: master, dev and a feature branch merged by a pull request.

Run from this folder: python generate_git_branches.py
"""

from pathlib import Path

WIDTH, HEIGHT = 760, 250
LANES = {"master": 50, "dev": 125, "feature": 200}
COLORS = {"master": "#2f6f9f", "dev": "#4f8a3a", "feature": "#c07a1a"}
LABELS = {
    "master": "master: released versions",
    "dev": "dev: next release",
    "feature": "feature/new-binding-model",
}

# Commits per lane (x positions)
COMMITS = {
    "master": [190, 650],
    "dev": [190, 260, 330, 560, 620],
    "feature": [390, 450, 510],
}

# Connections between lanes: (x_from, lane_from, x_to, lane_to, label)
LINKS = [
    (330, "dev", 390, "feature", "branch"),
    (510, "feature", 560, "dev", "pull request"),
    (620, "dev", 650, "master", "release"),
]


def svg():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="14">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker></defs>',
    ]
    for lane, y in LANES.items():
        xs = COMMITS[lane]
        parts.append(f'<text x="10" y="{y + 5}" fill="{COLORS[lane]}" font-weight="bold">{lane}</text>')
        parts.append(
            f'<line x1="{min(xs) - 30 if lane != "feature" else min(xs)}" y1="{y}" '
            f'x2="{max(xs) + (60 if lane != "feature" else 0)}" y2="{y}" '
            f'stroke="{COLORS[lane]}" stroke-width="4"/>'
        )
    for x1, lane1, x2, lane2, label in LINKS:
        y1, y2 = LANES[lane1], LANES[lane2]
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" '
            f'stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow)"/>'
        )
        parts.append(
            f'<text x="{(x1 + x2) / 2 + 8}" y="{(y1 + y2) / 2 + 5}" fill="#333">{label}</text>'
        )
    for lane, xs in COMMITS.items():
        for x in xs:
            parts.append(
                f'<circle cx="{x}" cy="{LANES[lane]}" r="9" fill="white" '
                f'stroke="{COLORS[lane]}" stroke-width="4"/>'
            )
    for lane, y in LANES.items():
        x = COMMITS[lane][0] - 20 if lane == "feature" else 100
        parts.append(f'<text x="{x}" y="{y + 32 if lane == "feature" else y - 16}" fill="#333">{LABELS[lane]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    Path(__file__).with_name("git_branches.svg").write_text(svg())
