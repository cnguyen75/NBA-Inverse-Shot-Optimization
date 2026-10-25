"""Draw the README figures from results/results.json and demo aggregates.

    python scripts/make_figures.py DEMO.json   # writes figures/*.svg

Hand-written SVG in the site style: card #F6F2E8, ink #231F1B, secondary
#3D372F, rule #B3A992; series blue #1F6FB5, orange #C0601F, purple #7A4E9E;
Archivo; direct labels, no legend.
"""

import json
import os
import sys
from xml.sax.saxutils import escape


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD, INK, SECONDARY, RULE = "#F6F2E8", "#231F1B", "#3D372F", "#B3A992"
BLUE, ORANGE, PURPLE, GREY = "#1F6FB5", "#C0601F", "#7A4E9E", "#8A8171"
NAMES = {
    "lebron james": "LeBron James",
    "stephen curry": "Stephen Curry",
    "james harden": "James Harden",
}


def head(W, H, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="t d" font-family="Archivo, sans-serif">',
        f'<title id="t">{escape(title)}</title>',
        f'<desc id="d">{escape(desc)}</desc>',
        f'<rect width="{W}" height="{H}" fill="{CARD}"/>',
    ]


def text(x, y, s, size=14, fill=INK, weight="400", anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def comparison_svg(res):
    chosen = f"model_{res['chosen_variant']}"
    rows = [
        ("majority_class", "Always “not chosen”", GREY),
        ("logistic_regression", "Logistic regression", PURPLE),
        ("model_repo", "Cost network, notebook's loss", ORANGE),
    ]
    if chosen != "model_repo":
        rows.append(
            (
                chosen,
                f"Cost network, {res['chosen_variant'].replace('_', ' ')} (chosen on validation)",
                BLUE,
            )
        )
    metrics = [
        ("accuracy", "Accuracy", 0.5, 1.0),
        ("roc_auc", "ROC AUC", 0.5, 1.0),
        ("top1_within_shot", "Real shot ranked first", 0.0, 1.0),
    ]
    W, label_w, col_w, gap = 1000, 290, 210, 26
    row_h, top, ph = 26, 112, 30
    players = list(res["players"])
    H = top + len(players) * (ph + len(rows) * row_h + 22) + 50
    p = head(
        W,
        H,
        "Held-out comparison: majority class, logistic regression and the cost network",
        "Dots are test-set values; lines are 95% bootstrap intervals over test shots. "
        "Source: results/results.json.",
    )
    p.append(
        text(
            24,
            34,
            "Does the cost network beat simple baselines on held-out shots?",
            20,
            weight="700",
        )
    )
    p.append(
        text(
            24,
            56,
            "Test split (20% of shots, never used for fitting or choices). Lines: 95% "
            "bootstrap interval over shots.",
            14,
            SECONDARY,
        )
    )
    for j, (_, title, lo, hi) in enumerate(metrics):
        x0 = label_w + j * (col_w + gap)
        p.append(text(x0, top - 26, title, 13.5, weight="700"))
        for t in (lo, (lo + hi) / 2, hi):
            p.append(
                text(
                    x0 + (t - lo) / (hi - lo) * col_w,
                    top - 8,
                    f"{t:.2f}".rstrip("0").rstrip("."),
                    11,
                    SECONDARY,
                    anchor="middle",
                )
            )
    y = top
    for pl in players:
        r = res["players"][pl]
        p.append(text(24, y + 18, NAMES[pl], 15, weight="700"))
        chance = r["test"]["logistic_regression"]["top1_chance"]
        y0 = y + ph
        for j, (key, _, lo, hi) in enumerate(metrics):
            x0 = label_w + j * (col_w + gap)
            for t in (lo, (lo + hi) / 2, hi):
                xx = x0 + (t - lo) / (hi - lo) * col_w
                p.append(
                    f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y0 - 10}" y2="{y0 + len(rows) * row_h - 12}" '
                    f'stroke="{RULE}" stroke-dasharray="3 3"/>'
                )
            if key == "top1_within_shot":
                xx = x0 + chance * col_w
                p.append(
                    f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y0 - 10}" '
                    f'y2="{y0 + len(rows) * row_h - 12}" stroke="{INK}" stroke-width="1.2"/>'
                )
                p.append(text(xx + 3, y0 - 13, f"chance {chance:.2f}", 10.5, SECONDARY))
        for i, (k, name, color) in enumerate(rows):
            cy = y0 + i * row_h
            p.append(text(24, cy + 5, name, 13))
            for j, (key, _, lo, hi) in enumerate(metrics):
                v = r["test"][k].get(key)
                if v is None:
                    continue
                x0 = label_w + j * (col_w + gap)
                sx = lambda t: x0 + (min(max(t, lo), hi) - lo) / (hi - lo) * col_w  # noqa: E731
                ci = r["test_ci95"].get(k, {}).get(key)
                if ci:
                    p.append(
                        f'<line x1="{sx(ci[0]):.1f}" x2="{sx(ci[1]):.1f}" y1="{cy}" y2="{cy}" '
                        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
                    )
                p.append(f'<circle cx="{sx(v):.1f}" cy="{cy}" r="5.5" fill="{color}"/>')
                p.append(text(sx(v), cy - 8, f"{v:.2f}", 10.5, SECONDARY, anchor="middle"))
        y = y0 + len(rows) * row_h + 22
    p.append(
        text(
            24,
            H - 20,
            "The majority class has no ranking, so it has no AUC interval or within-shot rank.",
            12,
            SECONDARY,
        )
    )
    p.append("</svg>")
    return "\n".join(p)


def importance_svg(demo):
    feats = list(demo["players"][0]["importance"]["values"])
    order = sorted(
        feats, key=lambda f: -sum(pl["importance"]["values"][f]["mean"] for pl in demo["players"])
    )
    W, label_w, col_w, gap = 1000, 170, 230, 30
    row_h, top = 24, 100
    H = top + len(order) * row_h + 60
    p = head(
        W,
        H,
        "Learned importance of each factor, by player",
        "Bars: mean |gradient| importance over 5 training seeds, normalised to sum to 1; "
        "lines: range across seeds. Source: results/results.json.",
    )
    p.append(text(24, 34, "Which factors move each player's learned score most?", 20, weight="700"))
    p.append(
        text(
            24,
            56,
            "Mean |gradient| of the chosen network's score per input, normalised to "
            "sum to 1. Bar: mean of 5 training seeds; line: their range.",
            14,
            SECONDARY,
        )
    )
    vmax = max(pl["importance"]["values"][f]["max"] for pl in demo["players"] for f in feats)
    colors = [BLUE, ORANGE, PURPLE]
    for j, pl in enumerate(demo["players"]):
        x0 = label_w + j * (col_w + gap)
        p.append(text(x0, top - 16, pl["name"], 14, weight="700"))
        for i, f in enumerate(order):
            v = pl["importance"]["values"][f]
            cy = top + i * row_h
            w = v["mean"] / vmax * (col_w - 50)
            p.append(
                f'<rect x="{x0}" y="{cy - 7}" width="{w:.1f}" height="12" fill="{colors[j]}"/>'
            )
            p.append(
                f'<line x1="{x0 + v["min"] / vmax * (col_w - 50):.1f}" '
                f'x2="{x0 + v["max"] / vmax * (col_w - 50):.1f}" y1="{cy + 9}" y2="{cy + 9}" '
                f'stroke="{INK}" stroke-width="1.2"/>'
            )
            p.append(text(x0 + w + 6, cy + 4, f"{v['mean']:.2f}", 11.5, SECONDARY))
    for i, f in enumerate(order):
        p.append(text(24, top + i * row_h + 4, f, 13))
    p.append("</svg>")
    return "\n".join(p)


def zones_svg(demo):
    W, top = 1000, 96
    band_h = 30
    zones = [z["label"] for z in demo["players"][0]["zones"]]
    H = top + len(demo["players"]) * (18 + len(zones) * (band_h - 6) + 34) - 10
    p = head(
        W,
        H,
        "Where each player shot from, 2014-15, by distance band",
        "Bars: share of the player's attempts in each distance band; labels: attempts and "
        "field-goal percentage. Source: shot logs, aggregated.",
    )
    p.append(text(24, 34, "Where each player shot from", 20, weight="700"))
    p.append(
        text(
            24,
            56,
            "Share of 2014–15 attempts by distance band (the data has distance, not "
            "court position), with field-goal percentage.",
            14,
            SECONDARY,
        )
    )
    colors = [BLUE, ORANGE, PURPLE]
    y = top
    for j, pl in enumerate(demo["players"]):
        p.append(
            text(
                24,
                y,
                f"{pl['name']}: {pl['attempts']:,} attempts, {pl['fg_pct'] * 100:.1f}% made",
                15,
                weight="700",
            )
        )
        y += 18
        for z in pl["zones"]:
            w = z["share_of_attempts"] * 560
            p.append(text(24, y + 15, z["label"], 13))
            p.append(f'<rect x="130" y="{y + 3}" width="{w:.1f}" height="16" fill="{colors[j]}"/>')
            p.append(
                text(
                    136 + w,
                    y + 16,
                    f"{z['share_of_attempts'] * 100:.0f}% of attempts · "
                    f"{z['attempts']} shots · {z['fg_pct'] * 100:.1f}% made · "
                    f"defender {z['median_defender_ft']} ft (median)",
                    12,
                    SECONDARY,
                )
            )
            y += band_h - 6
        y += 34
    p.append("</svg>")
    return "\n".join(p)


def main():
    res = json.load(open(os.path.join(ROOT, "results", "results.json")))
    demo = json.load(open(sys.argv[1]))
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    for name, svg in (
        ("comparison", comparison_svg(res)),
        ("importance", importance_svg(demo)),
        ("zones", zones_svg(demo)),
    ):
        with open(os.path.join(ROOT, "figures", f"{name}.svg"), "w") as f:
            f.write(svg + "\n")
        print(f"figures/{name}.svg")


if __name__ == "__main__":
    main()
