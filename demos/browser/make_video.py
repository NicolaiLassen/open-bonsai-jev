#!/usr/bin/env python3
"""
Composite the recorded run into a presentable video.

Playwright records the page viewport and nothing else, so the dashboard is
drawn afterwards rather than injected into the page: the run writes a timeline
(one entry per arrival and per decision, timestamped against the start of the
recording), this renders one panel image per timeline state, and ffmpeg lays
the panel over the video on a dark canvas.

Doing it in post rather than as an on-page overlay keeps the recording honest
-- the browser video is untouched, and the panel cannot slow the run down or
change what the page looked like.

    make_video.py results/wikigame-run.json
    make_video.py results/wikigame-run.json --gif
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

W, H = 1920, 1080          # output canvas
VX, VY, VW, VH = 56, 214, 1152, 720   # where the 1280x800 recording sits
PANEL_X = VX + VW + 48

PANEL_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:{W}px; height:{H}px; background:transparent; }}
  body {{ font-family:Inter,-apple-system,system-ui,sans-serif; color:#fff;
          -webkit-font-smoothing:antialiased; }}
  .mono {{ font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; }}
  .bg {{ position:absolute; background:#090908; }}
  .glow {{ position:absolute; left:{gx}px; top:{gy}px; width:{gw}px; height:{gh}px;
           border-radius:50%; filter:blur(150px); opacity:.30;
           background:radial-gradient(circle,#2a78d6 0%,transparent 70%); }}
  .glow2 {{ position:absolute; right:-140px; bottom:-160px; width:620px; height:620px;
            border-radius:50%; filter:blur(160px); opacity:.20;
            background:radial-gradient(circle,#eb6834 0%,transparent 70%); }}
  .ring {{ position:absolute; left:{rx}px; top:{ry}px; width:{rw}px; height:{rh}px;
           border:1px solid #33322d; border-radius:16px;
           box-shadow:0 30px 90px rgba(0,0,0,.7), 0 0 0 7px rgba(255,255,255,.022);
           pointer-events:none; }}
  .brand {{ position:absolute; left:60px; top:58px; display:flex;
            align-items:center; gap:14px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; background:{accent};
          box-shadow:0 0 22px {accent}; }}
  .brand h1 {{ font-size:29px; font-weight:800; letter-spacing:-.028em; }}
  .brand .sub {{ margin-left:4px; padding:5px 12px; border-radius:99px;
                 background:#161614; border:1px solid #2b2a26; color:#8fb7ee;
                 font-size:13.5px; font-weight:600; }}
  .kicker {{ position:absolute; right:60px; top:52px; text-align:right; }}
  .kicker .k1 {{ color:#5c5b55; font-size:12px; font-weight:700;
                 letter-spacing:.16em; text-transform:uppercase; }}
  .kicker .k2 {{ margin-top:7px; font-family:"JetBrains Mono",monospace;
                 font-size:19px; font-weight:700; color:#fff;
                 letter-spacing:-.01em; font-variant-numeric:tabular-nums; }}
  .rule {{ position:absolute; left:60px; right:60px; top:122px; height:1px;
           background:linear-gradient(90deg,#33322d,#1a1a17 70%,transparent); }}
  .p {{ position:absolute; left:{px}px; top:196px; width:{pw}px; }}
  .lbl {{ color:#5c5b55; font-size:12px; font-weight:700; text-transform:uppercase;
          letter-spacing:.16em; margin-bottom:12px; }}
  .goal {{ display:flex; align-items:center; flex-wrap:wrap; gap:10px;
           font-size:26px; font-weight:700; letter-spacing:-.018em; margin-bottom:38px; }}
  .goal .to {{ color:#45443f; font-size:22px; }}
  .goal .end {{ color:#eb6834; }}
  .stat {{ display:flex; align-items:flex-end; gap:34px; margin-bottom:12px; }}
  .big {{ font-family:"JetBrains Mono",monospace; font-size:88px; font-weight:700;
          line-height:.86; letter-spacing:-.055em; font-variant-numeric:tabular-nums; }}
  .big span {{ font-size:40px; color:#5c5b55; margin-left:6px;
                letter-spacing:0; }}
  .side {{ padding-bottom:9px; }}
  .side .v {{ font-family:"JetBrains Mono",monospace; font-size:24px; font-weight:700;
              color:#fff; font-variant-numeric:tabular-nums; white-space:nowrap;
              letter-spacing:-.02em; }}
  .side .v u {{ text-decoration:none; color:#5c5b55; font-size:16px; margin-left:3px; }}
  .side .k {{ color:#5c5b55; font-size:11.5px; font-weight:600; letter-spacing:.14em;
              text-transform:uppercase; margin-top:5px; }}
  .segs {{ display:flex; gap:5px; margin:26px 0 36px; }}
  .seg {{ height:6px; flex:1; border-radius:3px; background:#1f1e1b; }}
  .seg.on {{ background:linear-gradient(90deg,#2a78d6,#3987e5);
             box-shadow:0 0 12px #3987e577; }}
  .path {{ list-style:none; }}
  .path li {{ display:flex; align-items:center; gap:14px; padding:6px 0 6px 14px;
              border-left:2px solid #1f1e1b; }}
  .path li.gap {{ color:#3d3c37; font-family:"JetBrains Mono",monospace;
                  font-size:13px; padding:2px 0 2px 14px; }}
  .path li.now {{ border-left-color:#3987e5; }}
  .path li.done {{ border-left-color:#2b2a26; }}
  .n {{ color:#3d3c37; font-family:"JetBrains Mono",monospace; font-size:14px;
        font-weight:500; width:24px; }}
  .t {{ flex:1; font-size:18.5px; font-weight:500; color:#4e4d47;
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  li.done .t {{ color:#a9a89e; }}
  li.now .t {{ color:#fff; font-weight:700; }}
  .ms {{ font-family:"JetBrains Mono",monospace; font-size:14px; color:#3d3c37;
         font-variant-numeric:tabular-nums; }}
  li.done .ms {{ color:#5c5b55; }}
  li.now .ms {{ color:#3987e5; font-weight:700; }}
  .foot {{ position:absolute; left:{px}px; bottom:62px; width:{pw}px;
           padding:20px 22px; border-radius:14px; background:#0f0f0d;
           border:1px solid #23221e; }}
  .foot p {{ color:#8a897f; font-size:16px; line-height:1.62; }}
  .foot b {{ color:#fff; font-weight:600; }}
  .foot .hl {{ color:{accent}; font-weight:700; }}
</style></head><body>
  <div class="bg" style="left:0;top:0;width:{W}px;height:{VY}px"></div>
  <div class="bg" style="left:0;top:{vy2}px;width:{W}px;height:{h2}px"></div>
  <div class="bg" style="left:0;top:{VY}px;width:{VX}px;height:{VH}px"></div>
  <div class="bg" style="left:{vx2}px;top:{VY}px;width:{w2}px;height:{VH}px"></div>
  <div class="glow"></div><div class="glow2"></div>
  <div class="ring"></div>
  <div class="brand"><div class="dot"></div><h1>open-bonzi-jev</h1>
    <div class="sub">{model} &middot; {size} GB</div></div>
  <div class="kicker"><div class="k1">Wikipedia game</div><div class="k2">{label}</div></div>
  <div class="rule"></div>
  <div class="p">
    <div class="lbl">Goal</div>
    <div class="goal">{start}<span class="to">&rarr;</span><span class="end">{goal}</span></div>
    <div class="stat">
      <div class="big">{elapsed}<span>s</span></div>
      <div class="side"><div class="v">{hops}<u>/{total_hops}</u></div>
        <div class="k">hops</div></div>
      <div class="side"><div class="v">{last_ms}<u>ms</u></div>
        <div class="k">last decision</div></div>
    </div>
    <div class="segs">{segs}</div>
    <div class="lbl">Path</div>
    <ol class="path">{rows}</ol>
  </div>
  <div class="foot"><p>{caption}</p></div>
</body></html>"""


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_states(run: dict) -> list[dict]:
    """One panel state per timeline event, each with a start and end time."""
    tl = run["timeline"]
    path = run["path"]
    states = []
    for i, e in enumerate(tl):
        t_end = tl[i + 1]["t"] if i + 1 < len(tl) else run["video_end_s"]
        if t_end <= e["t"]:
            continue
        reached = path.index(e["title"]) + 1 if e["title"] in path else 0
        states.append({
            "start": e["t"], "end": t_end, "kind": e["kind"],
            "here": e["title"], "reached": reached,
            "ms": e.get("ms"), "choice": e.get("choice"),
        })
    return states


def panel_html(run: dict, st: dict, arrive_times: dict) -> str:
    path = run["path"]
    n = len(path)
    # Window the path around the current hop. A 20-article run will not fit,
    # and the tail is what matters while it is running.
    WINDOW = 11
    cur = max(1, st["reached"])
    if n <= WINDOW:
        lo, hi = 1, n
    else:
        lo = max(1, min(cur - WINDOW + 4, n - WINDOW + 1))
        hi = lo + WINDOW - 1
    rows = []
    if lo > 1:
        rows.append(f'<li class="gap">&vellip; {lo - 1} earlier</li>')
    for i in range(lo, hi + 1):
        title = path[i - 1]
        cls = "now" if i == st["reached"] else ("done" if i < st["reached"] else "")
        t = arrive_times.get(title)
        ms = f"{t:.1f}s" if (t is not None and i <= st["reached"]) else ""
        rows.append(f'<li class="{cls}"><span class="n">{i:02d}</span>'
                    f'<span class="t">{esc(title)}</span>'
                    f'<span class="ms">{ms}</span></li>')
    if hi < n:
        rows.append(f'<li class="gap">&vellip; {n - hi} more</li>')
    segs = "".join(f'<div class="seg{" on" if i < st["reached"] else ""}"></div>'
                   for i in range(n))

    if st["reached"] >= n and run.get("arrived"):
        cap = (f'<span class="hl">Reached {esc(run["goal"])} in '
               f'{run["clicks"]} clicks.</span> Each click was one forward pass '
               f'of a <b>{run["size_gb"]} GB</b> model at '
               f'<b>{run["ms_per_decision"]} ms</b>, so the model accounted for '
               f'only <b>{run["model_s"]:.1f}s</b> of {run["wall_s"]:.1f}s. '
               f'The rest is page loads.')
    elif st["kind"] == "decide":
        cap = (f'Read the links on this page and chose '
               f'<b>{esc(st["choice"] or "")}</b> in '
               f'<span class="hl">{st["ms"]} ms</span>. One forward pass; the '
               f'option letter with the highest probability <i>is</i> the click.')
    elif st["reached"] >= n and not run.get("arrived"):
        cap = (f'Ran out of moves after <b>{run["clicks"]} clicks</b> without '
               f'reaching {esc(run["goal"])}. The decisions were fast '
               f'(<b>{run["ms_per_decision"]} ms</b> each); the navigation '
               f'drifted.')
    else:
        cap = ('Landed. Next: score every link on this page and click the best '
               'one, in a single forward pass.')

    done = st["reached"] >= n and run.get("arrived")
    return PANEL_HTML.format(
        W=W, H=H, VX=VX, VY=VY, VW=VW, VH=VH,
        vx2=VX + VW, w2=W - VX - VW, vy2=VY + VH, h2=H - VY - VH,
        rx=VX - 1, ry=VY - 1, rw=VW + 2, rh=VH + 2,
        gx=PANEL_X - 120, gy=140, gw=640, gh=640,
        px=PANEL_X, pw=W - PANEL_X - 60,
        accent="#1baf7a" if done else "#eb6834",
        model=esc(run["model"]), size=run["size_gb"],
        start=esc(run["start"]), goal=esc(run["goal"]),
        elapsed=f'{st["end"]:.1f}', hops=max(0, st["reached"] - 1),
        total_hops=n - 1, last_ms=st.get("ms") or run["ms_per_decision"],
        segs=segs, rows="".join(rows), caption=cap,
        label=esc(run.get("_label", "")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="composite the run into a video")
    ap.add_argument("run", help="the --json file written by wikinav.py")
    ap.add_argument("--out", default=str(HERE / "wiki-game.mp4"))
    ap.add_argument("--gif", action="store_true", help="also write a compact GIF")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--label", help="headline shown top-right; defaults to "
                                    "size, clicks and wall time")
    args = ap.parse_args(argv)

    run = json.loads(Path(args.run).read_text())
    video = Path(run["video"])
    if not video.exists():
        raise SystemExit(f"recording not found: {video}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH")

    run["_label"] = args.label or (
        f'{run["size_gb"]} GB  ·  {run["clicks"]} clicks  ·  {run["wall_s"]:.1f} s'
        if run.get("arrived") else
        f'{run["size_gb"]} GB  ·  did not finish')
    arrive_times = {e["title"]: e["t"] for e in run["timeline"]
                    if e["kind"] == "arrive"}
    states = build_states(run)
    print(f"{len(states)} panel states over {run['video_end_s']}s")

    tmp = Path(tempfile.mkdtemp(prefix="bonzi-vid-"))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        pg = b.new_context(viewport={"width": W, "height": H},
                           device_scale_factor=1).new_page()
        for i, st in enumerate(states):
            pg.set_content(panel_html(run, st, arrive_times))
            pg.screenshot(path=str(tmp / f"p{i:03d}.png"), omit_background=True)
        b.close()

    # one overlay per state, switched on its time window
    inputs, filters, last = ["-i", str(video)], [], "[bg]"
    filters.append(f"color=c=#0b0b0a:s={W}x{H}:r={args.fps}[bg0]")
    filters.append(f"[0:v]scale={VW}:{VH},setpts=PTS-STARTPTS[vid]")
    filters.append(f"[bg0][vid]overlay={VX}:{VY}[bg]")
    for i, st in enumerate(states):
        inputs += ["-i", str(tmp / f"p{i:03d}.png")]
        nxt = f"[s{i}]"
        filters.append(
            f"{last}[{i+1}:v]overlay=0:0:enable='between(t,{st['start']:.2f},"
            f"{st['end']:.2f})'{nxt}")
        last = nxt
    graph = ";".join(filters)

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", last,
           "-r", str(args.fps), "-t", str(run["video_end_s"]),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-preset", "medium", args.out]
    print("compositing ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:], file=sys.stderr)
        raise SystemExit("ffmpeg failed")
    print("wrote", args.out)

    if args.gif:
        gif = str(Path(args.out).with_suffix(".gif"))
        pal = str(tmp / "pal.png")
        subprocess.run(["ffmpeg", "-y", "-i", args.out, "-vf",
                        "fps=12,scale=900:-1:flags=lanczos,palettegen=max_colors=128",
                        pal], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", args.out, "-i", pal, "-lavfi",
                        "fps=12,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse="
                        "dither=bayer:bayer_scale=3", gif], capture_output=True)
        print("wrote", gif, f"({Path(gif).stat().st_size/1e6:.1f} MB)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
