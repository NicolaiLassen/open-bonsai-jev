#!/usr/bin/env python3
"""
Render the image grid: each photo with the question it was asked and the
probability that came back.

Reads results/vision.json (written by vision_score.py --json) and screenshots
an HTML page, so the grid stays in step with the measurements rather than
being drawn by hand.

    make_grid.py            # -> demos/vision/vision-grid.png
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

CELL_W, IMG_H = 380, 250
COLS = 3

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{W}px; background:#0b0b0a; color:#fff; padding:34px 34px 26px;
          font-family:Inter,-apple-system,system-ui,sans-serif;
          -webkit-font-smoothing:antialiased; }}
  h1 {{ font-size:22px; font-weight:800; letter-spacing:-.02em; }}
  .sub {{ color:#8a897f; font-size:14px; margin-top:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat({COLS}, 1fr);
           gap:20px; margin-top:26px; }}
  .cell {{ background:#131312; border:1px solid #26251f; border-radius:14px;
           overflow:hidden; }}
  .ph {{ height:{IMG_H}px; background:#000; display:flex;
         align-items:center; justify-content:center; overflow:hidden; }}
  .ph img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .body {{ padding:13px 15px 15px; }}
  .q {{ font-size:15.5px; font-weight:650; }}
  .bar {{ height:9px; background:#26251f; border-radius:5px; margin:11px 0 8px;
          overflow:hidden; }}
  .fill {{ height:100%; border-radius:5px;
           background:linear-gradient(90deg,#15916590,#1baf7a); }}
  .row {{ display:flex; align-items:baseline; justify-content:space-between; }}
  .ans {{ font:700 13px JetBrains Mono,ui-monospace,monospace; color:#1baf7a;
          text-transform:uppercase; letter-spacing:.05em; }}
  .p {{ font:700 17px JetBrains Mono,ui-monospace,monospace;
        font-variant-numeric:tabular-nums; }}
  .foot {{ margin-top:24px; color:#6f6e68; font-size:12.5px; }}
</style></head><body>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
  <div class="grid">{cells}</div>
  <div class="foot">{foot}</div>
</body></html>"""

CELL = """<div class="cell">
  <div class="ph"><img src="data:image/jpeg;base64,{b64}"></div>
  <div class="body">
    <div class="q">{q}</div>
    <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
    <div class="row"><span class="ans">{ans}</span><span class="p">{p:.3f}</span></div>
  </div>
</div>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="render the vision result grid")
    ap.add_argument("--results", default=str(ROOT / "results" / "vision.json"))
    ap.add_argument("--out", default=str(HERE / "vision-grid.jpg"))
    args = ap.parse_args(argv)

    rows = [r for r in json.loads(Path(args.results).read_text())
            if r.get("task") == "matched"]
    if not rows:
        raise SystemExit("no matched-question rows in the results file")

    cells = []
    for r in rows:
        img = HERE / "images" / f"{r['image']}.jpg"
        p = r["probabilities"]["true"]
        cells.append(CELL.format(
            b64=base64.b64encode(img.read_bytes()).decode(),
            q=r["question"], pct=p * 100, p=p, ans=r["choice"]))

    med = sorted(r["forward_s"] for r in rows)[len(rows) // 2]
    html = PAGE.format(
        W=CELL_W * COLS + 20 * (COLS - 1) + 68, COLS=COLS, IMG_H=IMG_H,
        title="One forward pass per picture",
        subtitle="Bonsai 2 27B, 5.95 GB plus a 0.63 GB vision projector, "
                 "running on a laptop GPU. The answer is read off the "
                 "next-token distribution; nothing is generated.",
        cells="".join(cells),
        foot=f"{sum(r['correct'] for r in rows)}/{len(rows)} correct &middot; "
             f"median {med * 1000:.0f} ms per image &middot; "
             f"photos are the lead image of the matching Wikipedia article")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": CELL_W * COLS + 20 * (COLS - 1) + 68,
                                     "height": 400},
                           device_scale_factor=2).new_page()
        pg.set_content(html)
        pg.wait_for_timeout(900)
        # the cells are photographs, so JPEG rather than PNG: same
        # result at a fifth of the size in a README
        pg.screenshot(path=args.out, full_page=True,
                      type="jpeg", quality=88)
        b.close()
    print("wrote", args.out, f"({Path(args.out).stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
