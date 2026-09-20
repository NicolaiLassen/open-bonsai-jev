#!/usr/bin/env python3
"""
Render the benchmark charts as static SVG.

Static, because these are read inside README.md on GitHub and inside a model
card on HuggingFace, where scripts never run. Dark mode is handled with a
`prefers-color-scheme` block inside each SVG, so one file works on both themes.

    make_charts.py            # reads results/benchmark.json -> docs/charts/*.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"

# dataviz reference palette, slots 1 and 2, validated for both surfaces.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "ink3": "#84837d",
    "grid": "#e4e3df", "s1": "#2a78d6", "s2": "#eb6834",
    # ordinal ramp for the three difficulty tiers: one hue, light to dark,
    # starting no lighter than step 250 so it clears 2:1 on the light surface
    "t0": "#86b6ef", "t1": "#3987e5", "t2": "#1c5cab", "s3": "#1baf7a",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "ink3": "#8a897f",
    "grid": "#33322f", "s1": "#3987e5", "s2": "#d95926",
    "t0": "#9ec5f4", "t1": "#3987e5", "t2": "#184f95", "s3": "#199e70",
}


def style_block() -> str:
    def rules(d, sel):
        return "\n".join(f"    {sel} {{ --{k}: {v}; }}" for k, v in [("", "")][:0]) or (
            f"    {sel} {{ " + " ".join(f"--{k}:{v};" for k, v in d.items()) + " }"
        )
    return f"""<style>
{rules(LIGHT, ':root')}
    @media (prefers-color-scheme: dark) {{
{rules(DARK, ':root')}
    }}
    .bg {{ fill: var(--surface); }}
    .ttl {{ fill: var(--ink); font: 600 15px -apple-system, "Segoe UI", system-ui, sans-serif; }}
    .sub {{ fill: var(--ink2); font: 400 11.5px -apple-system, "Segoe UI", system-ui, sans-serif; }}
    .lab {{ fill: var(--ink); font: 400 12px -apple-system, "Segoe UI", system-ui, sans-serif; }}
    .val {{ fill: var(--ink); font: 600 12px ui-monospace, "SF Mono", Menlo, monospace; }}
    .tick {{ fill: var(--ink3); font: 400 10.5px ui-monospace, "SF Mono", Menlo, monospace; }}
    .note {{ fill: var(--ink3); font: 400 10.5px -apple-system, system-ui, sans-serif; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .axis {{ stroke: var(--ink3); stroke-width: 1; }}
    .ref  {{ stroke: var(--ink3); stroke-width: 1; stroke-dasharray: 4 3; }}
    .refh {{ stroke: var(--s3); stroke-width: 2; stroke-dasharray: 7 4; }}
    .noteh {{ fill: var(--s3); font: 600 11.5px -apple-system, system-ui, sans-serif; }}
    .g1 {{ fill: var(--s1); }}  .g2 {{ fill: var(--s2); }}
    .g0 {{ fill: var(--s3); }}
    .tier0 {{ fill: var(--t0); }} .tier1 {{ fill: var(--t1); }} .tier2 {{ fill: var(--t2); }}
    .g1s {{ stroke: var(--s1); }} .g2s {{ stroke: var(--s2); }}
    .g0s {{ stroke: var(--s3); }}
  </style>"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def legend(x, y, items) -> str:
    out, dx = [], 0
    for label, cls in items:
        out.append(f'<rect x="{x+dx}" y="{y-8}" width="10" height="10" rx="2.5" class="{cls}"/>')
        out.append(f'<text x="{x+dx+15}" y="{y+1}" class="sub">{esc(label)}</text>')
        dx += 22 + int(len(label) * 6.1)
    return "".join(out)


def svg_open(w, h, title, subtitle):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">\n'
        f'  {style_block()}\n'
        f'  <rect class="bg" width="{w}" height="{h}" rx="8"/>\n'
        f'  <text x="24" y="30" class="ttl">{esc(title)}</text>\n'
        f'  <text x="24" y="48" class="sub">{esc(subtitle)}</text>\n'
    )


def bar_chart(models, path, eval_key, title, subtitle, chance=None, reference=None):
    """Horizontal bars. Length from zero, so linear scale only."""
    rows = [(m, m["evals"][eval_key]) for m in models if "accuracy" in m["evals"].get(eval_key, {})]
    left, right, top, rowh = 186, 74, 98, 30
    w = 720
    h = top + rowh * len(rows) + 62
    plot_w = w - left - right

    s = svg_open(w, h, title, subtitle)
    items = [("Bonsai 1", "g1"), ("Bonsai 2", "g2")]
    if any(m["generation"] == 0 for m, _ in rows):
        items.append(("Jev (hosted)", "g0"))
    s += legend(left, 66, items)

    for frac in (0, .25, .5, .75, 1.0):
        x = left + plot_w * frac
        s += f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + rowh*len(rows)}"/>'
        s += f'<text x="{x:.1f}" y="{top + rowh*len(rows) + 16}" class="tick" text-anchor="middle">{frac*100:.0f}%</text>'

    for i, (m, r) in enumerate(rows):
        y = top + i * rowh
        acc = r["accuracy"]
        bw = max(plot_w * acc, 3)
        cls = {0: "g0", 2: "g2"}.get(m["generation"], "g1")
        s += f'<text x="{left-12}" y="{y+19}" class="lab" text-anchor="end">{esc(m["label"])}</text>'
        # 4px rounded data-end, anchored at the zero baseline
        s += (f'<path class="{cls}" d="M{left} {y+7} H{left+bw-4} a4 4 0 0 1 4 4 v6 '
              f'a4 4 0 0 1 -4 4 H{left} Z"/>')
        s += f'<text x="{left+bw+8}" y="{y+19}" class="val">{acc*100:.1f}%</text>'

    base = top + rowh * len(rows)
    s += f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{base}"/>'

    notes = []
    if chance is not None:
        x = left + plot_w * chance
        s += f'<line class="ref" x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{base}"/>'
        s += f'<text x="{x:.1f}" y="{top-10}" class="note" text-anchor="middle">chance {chance*100:.0f}%</text>'
    if reference is not None:
        val, lab = reference
        x = left + plot_w * val
        s += f'<line class="ref" x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{base}"/>'
        s += f'<text x="{x:.1f}" y="{top-10}" class="note" text-anchor="middle">{esc(lab)}</text>'
        notes.append(lab)
    s += f'<text x="24" y="{h-14}" class="note">Higher is better. Measured on this repo&#8217;s eval file; see README for method.</text>'
    s += "</svg>\n"
    path.write_text(s, encoding="utf-8")
    return path


def dot_chart(models, path, eval_key, title, subtitle):
    """Throughput spans ~50x, so this is a dot plot on a log axis -- a bar would
    imply length-from-zero on a log scale, which is meaningless."""
    import math
    rows = [(m, m["evals"][eval_key]) for m in models
            if m["evals"].get(eval_key, {}).get("questions_per_min")]
    left, right, top, rowh = 186, 74, 76, 30
    w = 720
    h = top + rowh * len(rows) + 62
    plot_w = w - left - right
    vals = [r["questions_per_min"] for _, r in rows]
    lo, hi = 10, 1600
    def X(v):
        return left + plot_w * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    s = svg_open(w, h, title, subtitle)
    items = [("Bonsai 1", "g1"), ("Bonsai 2", "g2")]
    if any(m["generation"] == 0 for m, _ in rows):
        items.append(("Jev (hosted)", "g0"))
    s += legend(left, 62, items)
    for t in (10, 30, 100, 300, 1000):
        x = X(t)
        s += f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+rowh*len(rows)}"/>'
        s += f'<text x="{x:.1f}" y="{top+rowh*len(rows)+16}" class="tick" text-anchor="middle">{t}</text>'

    for i, (m, r) in enumerate(rows):
        y = top + i * rowh + 15
        v = r["questions_per_min"]
        cls = {0: "g0", 2: "g2"}.get(m["generation"], "g1")
        s += f'<text x="{left-12}" y="{y+4}" class="lab" text-anchor="end">{esc(m["label"])}</text>'
        s += f'<line class="{cls}s" x1="{X(lo)}" y1="{y}" x2="{X(v):.1f}" y2="{y}" stroke-width="2" opacity="0.32"/>'
        s += f'<circle class="{cls}" cx="{X(v):.1f}" cy="{y}" r="5.5"/>'
        s += f'<text x="{X(v)+12:.1f}" y="{y+4}" class="val">{v:.0f}</text>'
    s += (f'<text x="{left+plot_w/2:.1f}" y="{h-30}" class="sub" text-anchor="middle">'
          f'decisions per minute (log scale)</text>')
    s += (f'<text x="24" y="{h-14}" class="note">Higher is better. Warmup excluded. '
          f'Local models are one loaded model on this machine; Jev is a hosted API, '
          f'so its rate includes a network round trip.</text>')
    s += "</svg>\n"
    path.write_text(s, encoding="utf-8")
    return path


def scatter_chart(models, path, eval_key, title, subtitle):
    rows = [(m, m["evals"][eval_key]) for m in models
            if "accuracy" in m["evals"].get(eval_key, {}) and m.get("size_gb")]
    w, h = 720, 430
    left, right, top, bottom = 68, 30, 76, 66
    pw, ph = w - left - right, h - top - bottom
    xmax = max(m["size_gb"] for m, _ in rows) * 1.12
    ys = [r["accuracy"] for _, r in rows] + [
        m["evals"][eval_key]["accuracy"] for m in models
        if m["generation"] == 0 and "accuracy" in m["evals"].get(eval_key, {})]
    ylo, yhi = max(0, min(ys) - .08), min(1, max(ys) + .08)

    def X(v): return left + pw * (v / xmax)
    def Y(v): return top + ph * (1 - (v - ylo) / (yhi - ylo))

    s = svg_open(w, h, title, subtitle)
    hosted = [(m, m["evals"][eval_key]) for m in models
              if m["generation"] == 0 and "accuracy" in m["evals"].get(eval_key, {})]
    items = [("Bonsai 1", "g1"), ("Bonsai 2", "g2")]
    if hosted:
        items.append(("Jev (hosted, no local weights)", "g0"))
    s += legend(left, 62, items)
    for i in range(5):
        frac = i / 4
        yv = ylo + (yhi - ylo) * frac
        y = Y(yv)
        s += f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>'
        s += f'<text x="{left-10}" y="{y+4:.1f}" class="tick" text-anchor="end">{yv*100:.0f}%</text>'
    for i in range(5):
        xv = xmax * i / 4
        x = X(xv)
        s += f'<text x="{x:.1f}" y="{top+ph+18}" class="tick" text-anchor="middle">{xv:.1f}</text>'
    s += f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>'

    for m, r in rows:
        cls = {0: "g0", 2: "g2"}.get(m["generation"], "g1")
        x, y = X(m["size_gb"]), Y(r["accuracy"])
        # 2px surface ring keeps overlapping marks separable
        s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="var(--surface)"/>'
        s += f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
        anchor, dx = ("start", 12) if x < left + pw * .72 else ("end", -12)
        s += f'<text x="{x+dx:.1f}" y="{y+4:.1f}" class="lab" text-anchor="{anchor}">{esc(m["label"])}</text>'

    # Jev has no weights on disk, so it cannot take an x position; it is drawn
    # as the line to beat instead.
    for m, r in hosted:
        y = Y(r["accuracy"])
        if ylo <= r["accuracy"] <= yhi:
            s += (f'<line class="refh" x1="{left}" y1="{y:.1f}" '
                  f'x2="{left+pw}" y2="{y:.1f}"/>')
            s += (f'<text x="{left+pw-4}" y="{y-9:.1f}" class="noteh" '
                  f'text-anchor="end">{esc(m["label"])} {r["accuracy"]*100:.1f}%, '
                  f'no local weights</text>')
    s += f'<text x="{left+pw/2:.1f}" y="{h-26}" class="sub" text-anchor="middle">weights on disk (GB)</text>'
    s += f'<text x="24" y="{h-8}" class="note">Up and to the left is better: more accuracy per gigabyte.</text>'
    s += "</svg>\n"
    path.write_text(s, encoding="utf-8")
    return path



def wikigame_chart(path, data):
    """Solve rate by difficulty tier, grouped bars.

    Three tiers per model, so the shape of the failure is visible: everything
    solves the near tier, and the far tier is where they separate (or do not).
    """
    display = {"Bonsai-1.7B": "Bonsai 1 1.7B", "Bonsai-4B": "Bonsai 1 4B",
               "Bonsai-8B": "Bonsai 1 8B", "Bonsai-27B": "Bonsai 1 27B",
               "Ternary-Bonsai-8B": "Ternary Bonsai 1 8B",
               "Ternary-Bonsai-2-27B": "Bonsai 2 27B", "jev": "Jev (hosted)"}
    order = ["Bonsai-1.7B", "Bonsai-4B", "Bonsai-8B", "Bonsai-27B",
             "Ternary-Bonsai-8B", "Ternary-Bonsai-2-27B", "jev"]
    models = sorted(data["models"], key=lambda m: order.index(m)
                    if m in order else 99)
    tiers = ["near", "mid", "far"]
    tier_label = {"near": "near (1\u20133 hops)", "mid": "mid (same domain)",
                  "far": "far (cross-domain)"}

    stats = {}
    for m in models:
        rows = data["models"][m]["rows"]
        stats[m] = {}
        for t in tiers:
            sub = [r for r in rows if r.get("tier") == t]
            solved = [r for r in sub if r.get("arrived")]
            stats[m][t] = (len(solved), len(sub))

    w = 830
    left, right, top = 218, 118, 108
    grouph, barh, gap = 96, 22, 6
    h = top + grouph * len(models) + 66
    plot_w = w - left - right

    total_solved = {m: sum(v[0] for v in stats[m].values()) for m in models}
    total_n = sum(v[1] for v in stats[models[0]].values())

    s = svg_open(w, h, "Wikipedia game: solve rate by difficulty",
                 f"{total_n} curated start\u2192goal pairs, 20-click cap, "
                 f"links only. Higher is better.")
    s += legend(left, 72, [(tier_label[t], f"tier{i}") for i, t in enumerate(tiers)])

    for frac in (0, .25, .5, .75, 1.0):
        x = left + plot_w * frac
        s += (f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
              f'y2="{top + grouph * len(models) - 18}"/>')
        s += (f'<text x="{x:.1f}" y="{top + grouph * len(models) - 2}" class="tick" '
              f'text-anchor="middle">{frac * 100:.0f}%</text>')

    for mi, m in enumerate(models):
        gy = top + mi * grouph
        size = data["models"][m]["size_gb"]
        s += (f'<text x="{left - 14}" y="{gy + 30}" class="lab" '
              f'text-anchor="end">{esc(display.get(m, m))}</text>')
        where = f"{size} GB local" if size else "hosted API"
        s += (f'<text x="{left - 14}" y="{gy + 47}" class="sub" '
              f'text-anchor="end">{where} &#183; '
              f'{total_solved[m]}/{total_n} overall</text>')
        for ti, t in enumerate(tiers):
            solved, n = stats[m][t]
            frac = solved / n if n else 0
            y = gy + 8 + ti * (barh + gap)
            bw = max(plot_w * frac, 3)
            s += (f'<path class="tier{ti}" d="M{left} {y} H{left + bw - 4} '
                  f'a4 4 0 0 1 4 4 v{barh - 8} a4 4 0 0 1 -4 4 H{left} Z"/>')
            s += (f'<text x="{left + bw + 9}" y="{y + barh - 6}" class="val">'
                  f'{solved}/{n}</text>')
        s += f'<line class="axis" x1="{left}" y1="{gy + 4}" x2="{left}" y2="{gy + 4 + 3 * barh + 2 * gap}"/>'

    s += (f'<text x="24" y="{h - 14}" class="note">One deterministic run per '
          f'pair (argmax, temperature 0). Pairs and raw results are in the repo.</text>')
    s += "</svg>\n"
    path.write_text(s, encoding="utf-8")
    return path



def reliability_chart(path, cal):
    """Claimed probability against observed accuracy.

    The diagonal is perfect calibration; below it means the model claims more
    than it delivers. Bins holding fewer than MIN_N predictions are drawn but
    not connected, because a bin with two samples in it is noise and a line
    through it reads as signal: Jev's lowest bin is 2 predictions at 0%
    accuracy, which would otherwise drag its curve to the floor.
    """
    MIN_N = 5
    models = cal["models"]
    show = [("Ternary-Bonsai-2-27B", "raw", "g2", "Bonsai 2 27B, raw"),
            ("Ternary-Bonsai-2-27B", "cal", "g1", "Bonsai 2 27B, T=2.45"),
            ("jev", "raw", "g0", "Jev, raw")]
    show = [x for x in show if x[0] in models]

    # Both axes are probabilities, so the plot must be SQUARE: otherwise the
    # perfect-calibration diagonal is not at 45 degrees and the whole chart
    # reads as squashed. Width matches the other charts in the set.
    w = 720
    left, right, top, bottom = 68, 252, 92, 66
    pw = w - left - right
    ph = pw
    h = top + ph + bottom

    def X(v): return left + pw * v
    def Y(v): return top + ph * (1 - v)

    s = svg_open(w, h, "Do the probabilities mean anything?",
                 "What the model claimed, against how often it was right. "
                 "WANLI-256.")

    for i in range(5):
        v = i / 4
        s += f'<line class="grid" x1="{left}" y1="{Y(v):.1f}" x2="{left+pw}" y2="{Y(v):.1f}"/>'
        s += (f'<text x="{left-10}" y="{Y(v)+4:.1f}" class="tick" '
              f'text-anchor="end">{v*100:.0f}%</text>')
        s += (f'<text x="{X(v):.1f}" y="{top+ph+18}" class="tick" '
              f'text-anchor="middle">{v*100:.0f}%</text>')

    s += f'<line class="ref" x1="{X(0)}" y1="{Y(0):.1f}" x2="{X(1)}" y2="{Y(1):.1f}"/>'
    s += (f'<text x="{X(0.60):.1f}" y="{Y(0.60)-9:.1f}" class="note" '
          f'transform="rotate(-45 {X(0.60):.1f} {Y(0.60)-9:.1f})" '
          f'text-anchor="middle">perfectly calibrated</text>')
    s += (f'<text x="{X(0.97):.1f}" y="{Y(0.10):.1f}" class="note" '
          f'text-anchor="end">below the line = overconfident</text>')

    ly = top + 6
    for key, kind, cls, label in show:
        m = models[key]
        bins = m["reliability_raw" if kind == "raw" else "reliability_calibrated"]
        pts = [(b["confidence"], b["accuracy"], b["n"]) for b in bins
               if b["n"] and b["confidence"] is not None]
        solid = [p_ for p_ in pts if p_[2] >= MIN_N]
        if len(solid) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f} {Y(a):.1f}"
                         for i, (c, a, _) in enumerate(solid))
            s += f'<path d="{d}" fill="none" class="{cls}s" stroke-width="2.5"/>'
        for c, a, n in pts:
            r = max(3.5, min(9.0, 2.2 + (n ** 0.5) * 0.6))
            faint = ' opacity="0.35"' if n < MIN_N else ''
            s += (f'<circle cx="{X(c):.1f}" cy="{Y(a):.1f}" r="{r+1.5:.1f}" '
                  f'fill="var(--surface)"{faint}/>')
            s += (f'<circle class="{cls}" cx="{X(c):.1f}" cy="{Y(a):.1f}" '
                  f'r="{r:.1f}"{faint}/>')

        ece = m["ece_raw" if kind == "raw" else "ece_calibrated"]
        s += f'<circle class="{cls}" cx="{left+pw+26}" cy="{ly-4}" r="5.5"/>'
        s += f'<text x="{left+pw+40}" y="{ly}" class="lab">{esc(label)}</text>'
        s += (f'<text x="{left+pw+40}" y="{ly+17}" class="sub">'
              f'calibration error {ece:.3f}</text>')
        ly += 46

    s += (f'<text x="{left+pw+26}" y="{ly+6}" class="note">'
          f'dot size = predictions in that bin</text>')
    s += (f'<text x="{left+pw+26}" y="{ly+22}" class="note">'
          f'faded = fewer than {MIN_N}, not joined</text>')

    s += f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>'
    s += f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>'
    s += (f'<text x="{left+pw/2:.1f}" y="{h-30}" class="sub" '
          f'text-anchor="middle">probability the model claimed</text>')
    s += (f'<text x="24" y="{h-10}" class="note">Calibrating flattens the curve '
          f'onto the diagonal and stops it claiming 0.99 at all. The answers '
          f'themselves do not change.</text>')
    s += "</svg>\n"
    path.write_text(s, encoding="utf-8")
    return path


def main():
    src = ROOT / "results" / "benchmark.json"
    if not src.exists():
        raise SystemExit(f"no {src}; run scripts/benchmark.py first")
    data = json.loads(src.read_text())
    models = [m for m in data["models"] if m.get("evals")]
    OUT.mkdir(parents=True, exist_ok=True)

    made = [
        bar_chart(models, OUT / "wanli256-accuracy.svg", "wanli256",
                  "Decision quality: WANLI-256",
                  "Balanced 3-way natural-language inference. One forward pass per question.",
                  chance=1/3,
                  reference=(0.637, "SemIf Qwen3.5-4B 63.7%")),
        bar_chart(models, OUT / "easy100-accuracy.svg", "easy100",
                  "Sanity check: easy100",
                  "100 trivial true/false facts. A smoke test, not a benchmark; it saturates.",
                  chance=0.5),
        dot_chart(models, OUT / "throughput.svg", "wanli256",
                  "Throughput on WANLI-256",
                  "Apple M4 Pro, GPU (Metal). Model stays loaded; warmup excluded."),
        scatter_chart(models, OUT / "accuracy-vs-size.svg", "wanli256",
                      "Accuracy per gigabyte",
                      "WANLI-256 accuracy against weights on disk."),
    ]
    cal = ROOT / "results" / "calibration.json"
    if cal.exists():
        made.append(reliability_chart(OUT / "reliability.svg",
                                      json.loads(cal.read_text())))

    wg = ROOT / "results" / "wikigame-sweep110.json"
    if wg.exists():
        d = json.loads(wg.read_text())
        if d.get("models"):
            made.append(wikigame_chart(OUT / "wikigame-solve-rate.svg", d))

    for p in made:
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
