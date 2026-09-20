#!/usr/bin/env python3
"""
The Wikipedia game with a small Bonsai model: one forward pass per click.

Reproduces the demo from
[ts-browser-agent](https://github.com/ndrezn/ts-browser-agent) (LangChain +
TypeSafe Jev), which is itself the
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) pattern: reach
one Wikipedia article from another by clicking article links only. Their
default run is LangChain -> Microphone.

The whole agent is one typed decision repeated:

    A. Large language model   B. Microsoft   C. Software framework
    D. Prompt engineering     E. SCROLL_DOWN

Nothing is generated and nothing is parsed. The option letter with the highest
probability is the click.

The rules are enforced by **removing options, not by instructing**. Visited
pages, non-article links and namespace pages never reach the model. That is
ts-browser-agent's finding and it matched ours exactly: told "do not go back"
the model goes back anyway, and an offered DONE gets chosen on step one at
p=0.88.

    wikinav.py                                      # LangChain -> Microphone
    wikinav.py --start "Jimmy Page" --goal Microphone --model Bonsai-8B --video
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "openjev"))

from score import LETTERS, LlamaCppBackend, softmax_over, score_with  # noqa: E402

MAX_LINKS = 24
SCROLL = "SCROLL_DOWN"


def wiki_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def title_of(url: str) -> str:
    p = urlparse(url).path
    return unquote(p.split("/wiki/", 1)[1]).replace("_", " ") if "/wiki/" in p else url


# Wikipedia serves Parsoid HTML: article links are ABSOLUTE
# (href="https://en.wikipedia.org/wiki/Foo"), not relative /wiki/Foo, and the
# body is under #bodyContent, not the near-empty #mw-content-text.
SNAPSHOT_JS = r"""
() => {
  const root = document.querySelector('#bodyContent') || document.body;
  const links = [];
  const seen = new Set();
  for (const a of root.querySelectorAll('a[href]')) {
    const raw = a.getAttribute('href') || '';
    if (raw.startsWith('#')) continue;
    const i = raw.indexOf('/wiki/');
    if (i === -1) continue;
    if (raw.startsWith('http') && !/^https?:\/\/en\.wikipedia\.org\//.test(raw)) continue;
    if (raw.includes('?')) continue;                       // redirect=no stubs
    const t = raw.slice(i + 6).split('#')[0];
    if (!t || t.includes(':')) continue;                   // any namespace page
    // Disambiguation pages are navigational dead ends and a reliable trap:
    // one run went Shrink wrap -> Shrink Rap -> a TV series and never came
    // back. They are a rule of the game, so remove them rather than ask.
    if (/\(disambiguation\)/i.test(t)) continue;
    const text = (a.textContent || '').trim().replace(/\s+/g, ' ');
    if (!text || text.length > 60) continue;
    const r = a.getBoundingClientRect();
    // On screen, or within one screen below it: one scroll's worth of lookahead.
    if (!(r.width > 0 && r.height > 0 && r.bottom > 0 &&
          r.top < window.innerHeight * 2)) continue;
    const title = decodeURIComponent(t).replace(/_/g, ' ');
    const key = title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    links.push({ text, title, y: r.top });
  }
  links.sort((a, b) => a.y - b.y);
  const more = window.scrollY + window.innerHeight <
               document.documentElement.scrollHeight - 2;
  return { links, more,
           title: (document.querySelector('#firstHeading') || {}).textContent || '' };
}
"""



def pick_server(model_dir: Path) -> Path:
    from ggufinfo import needs_prism_runtime, read_gguf_info

    gguf = next((g for g in sorted(model_dir.glob("*.gguf"))
                 if "mmproj" not in g.name.lower()), None)
    if gguf is None:
        raise SystemExit(f"no .gguf in {model_dir}")
    fork = ROOT / "bonzi" / "llama.cpp-prism" / "build" / "bin" / "llama-server"
    stock = ROOT / "bonzi" / "llama.cpp" / "llama-server"
    if needs_prism_runtime(read_gguf_info(gguf)):
        if not fork.exists():
            raise SystemExit(
                f"{gguf.name} needs PrismML's llama.cpp; run "
                f"scripts/build_llama_fork.sh")
        return fork
    return stock if stock.exists() else fork


def run(args) -> int:
    from playwright.sync_api import sync_playwright

    # Pick the runtime from the weights, not from whichever binary exists:
    # Bonsai 2's ternary tensors need PrismML's build and stock refuses them.
    server = pick_server(ROOT / "bonzi" / "models" / args.model)

    mdir = ROOT / "bonzi" / "models" / args.model
    size = sum(f.stat().st_size for f in mdir.glob("*.gguf")) / 1e9
    print(f"loading {args.model} ({size:.2f} GB) ...", file=sys.stderr)
    t = time.perf_counter()
    backend = LlamaCppBackend(mdir, gpu=True, ctx=4096, llama_server=str(server))
    print(f"loaded in {time.perf_counter() - t:.1f}s\n", file=sys.stderr)

    target = args.goal.lower()
    visited: set[str] = set()
    path, timings, clicks = [], [], []
    arrived = False
    scrolls = 0
    t0 = time.perf_counter()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        kw = {"viewport": {"width": 1280, "height": 800}}
        if args.video:
            (HERE / "video").mkdir(exist_ok=True)
            kw["record_video_dir"] = str(HERE / "video")
            kw["record_video_size"] = {"width": 1280, "height": 800}
        ctx = browser.new_context(**kw)
        # Wikimedia's fundraising banner is served conditionally and covers the
        # top of the article, which both obscures the recording and pushes real
        # links out of view. Block the loader and hide the containers.
        ctx.route("**/*", lambda route: (
            route.abort() if "BannerLoader" in route.request.url
            or "centralnotice" in route.request.url.lower()
            else route.continue_()))
        ctx.add_init_script("""
          const css = `#siteNotice,#centralNotice,#frbanner,.frbanner,
            [id^="frb"],[class^="frb"],.cn-fundraising,.mw-dismissable-notice,
            #mw-indicator-fundraising { display:none !important; }`;
          const add = () => {
            if (document.getElementById('bonzi-nobanner')) return;
            const st = document.createElement('style');
            st.id = 'bonzi-nobanner';
            st.textContent = css;
            (document.head || document.documentElement).appendChild(st);
          };
          add();
          document.addEventListener('DOMContentLoaded', add);
        """)
        page = ctx.new_page()
        # The recording begins with the context, so every timeline entry is
        # measured from here and lines up with the video frames.
        t_video = time.perf_counter()
        timeline = []
        page.goto(wiki_url(args.start), wait_until="domcontentloaded")

        for step in range(1, args.max_steps + 1):
            page.wait_for_timeout(args.settle_ms)
            snap = page.evaluate(SNAPSHOT_JS)
            here = snap["title"] or title_of(page.url)
            if not path or path[-1] != here:
                path.append(here)
                timeline.append({"t": round(time.perf_counter() - t_video, 2),
                                 "kind": "arrive", "title": here,
                                 "hop": len(path) - 1})
            visited.add(here.lower())
            if here.lower() == target:
                arrived = True
                if args.hud:
                    try:
                        page.evaluate(HUD_JS, {
                            "model": args.model, "size": f"{size:.2f}",
                            "start": args.start, "goal": args.goal,
                            "ms": round(timings[-1] * 1000) if timings else 0,
                            "total_ms": round(sum(timings) * 1000),
                            "top": [], "step": len(clicks),
                            "options": 0, "path": path[-5:],
                            "history": [round(x * 1000) for x in timings[-14:]],
                            "arrived": True})
                        page.wait_for_timeout(max(args.dwell, 2500))
                    except Exception:
                        pass
                break

            cands = [l for l in snap["links"] if l["title"].lower() not in visited]
            options = [l["text"] for l in cands[:MAX_LINKS]]
            # Offer SCROLL_DOWN only when there is more page and we have not
            # just used it twice; unbounded, the model scrolls forever.
            if snap["more"] and scrolls < 2:
                options.append(SCROLL)
            if not options:
                print("  no options left")
                break

            state = {"goal": f"Reach the Wikipedia article: {args.goal}",
                     "current_article": here,
                     "articles_already_visited": path[-6:]}
            # Lead with the goal. ts-browser-agent found the classifier weighs
            # the opening of the question heavily, and a question that opens
            # with the current page reads as anchored to it.
            q = (f"Goal: reach the Wikipedia article {args.goal!r}. You win by "
                 f"using the fewest clicks, so pick the link most likely to get "
                 f"to {args.goal!r} fastest. Which of these links on {here!r} "
                 f"goes closest to {args.goal!r}?")
            r = score_with(backend, q, options, state)
            timings.append(r["forward_s"])
            probs = {options[LETTERS.index(l)]: p
                     for l, p in softmax_over(r["logits"]).items()}
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            choice = top[0][0]

            timeline.append({"t": round(time.perf_counter() - t_video, 2),
                             "kind": "decide", "title": here, "choice": choice,
                             "ms": round(r["forward_s"] * 1000),
                             "options": len(options),
                             "top": [[t_, round(float(p_), 3)] for t_, p_ in top]})
            print(f"  {step:2d}. {here[:30]:30s} -> {choice[:26]:26s} "
                  f"p={top[0][1]:.2f}  {r['forward_s']*1000:4.0f} ms "
                  f"({len(options)} options)")

            if args.hud:
                try:
                    page.evaluate(HUD_JS, {
                        "model": args.model, "size": f"{size:.2f}",
                        "start": args.start, "goal": args.goal,
                        "ms": round(r["forward_s"] * 1000),
                        "total_ms": round(sum(timings) * 1000),
                        "top": [[t_, float(p_)] for t_, p_ in top],
                        "step": len(clicks) + 1, "expected": args.expected_clicks,
                        "options": len(options),
                        "path": path[-5:],
                        "history": [round(x*1000) for x in timings[-14:]],
                        "arrived": False})
                    page.wait_for_timeout(args.dwell)
                except Exception:
                    pass

            if choice == SCROLL:
                scrolls += 1
                page.mouse.wheel(0, 800)
                continue
            scrolls = 0
            link = next(l for l in cands if l["text"] == choice)
            clicks.append({"from": here, "to": link["title"]})
            page.goto(wiki_url(link["title"]), wait_until="domcontentloaded")

        total = time.perf_counter() - t0
        video_end = time.perf_counter() - t_video
        vpath = page.video.path() if args.video else None
        ctx.close()
        browser.close()
    backend.close()

    n, model_s = len(timings), sum(timings)
    print()
    print(f"  {'REACHED' if arrived else 'did not reach'} {args.goal!r} in "
          f"{len(clicks)} clicks, {n} decisions")
    print(f"  wall clock       {total:.1f} s")
    if n:
        print(f"  model time       {model_s:.2f} s "
              f"({model_s/n*1000:.0f} ms per decision, "
              f"{model_s/total*100:.0f}% of wall clock)")
    print(f"  path             {' -> '.join(path)}")
    if vpath:
        print(f"  video            {vpath}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": args.model, "size_gb": round(size, 2),
            "start": args.start, "goal": args.goal, "arrived": arrived,
            "path": path, "clicks": len(clicks), "decisions": n,
            "wall_s": round(total, 2), "model_s": round(model_s, 2),
            "ms_per_decision": round(model_s / n * 1000) if n else None,
            "video": str(vpath) if vpath else None,
            "video_end_s": round(video_end, 2),
            "timeline": timeline,
        }, indent=2))
    return 0 if arrived else 1


HUD_JS = r"""
(d) => {
  document.getElementById('bonzi-hud')?.remove();
  const el = document.createElement('div');
  el.id = 'bonzi-hud';

  const BLUE = '#3987e5', ORANGE = '#eb6834', DIM = '#6f6e68', LINE = '#2a2a27';

  const row = ([name, p], i) => `
    <div style="display:flex;align-items:center;gap:10px;margin:6px 0;
                ${i ? 'opacity:.72' : ''}">
      <div style="width:24px;height:24px;border-radius:7px;flex:none;
                  display:flex;align-items:center;justify-content:center;
                  font:700 12px ui-monospace,Menlo,monospace;
                  background:${i ? '#232320' : BLUE};color:${i ? DIM : '#fff'};
                  ${i ? '' : `box-shadow:0 0 16px ${BLUE}66`}">
        ${String.fromCharCode(65 + i)}</div>
      <div style="width:220px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                  font:${i ? 400 : 650} 13.5px -apple-system,system-ui,sans-serif;
                  color:${i ? '#b8b7ad' : '#fff'}">${name}</div>
      <div style="flex:1;height:11px;background:#232320;border-radius:6px;overflow:hidden">
        <div style="width:${(p * 100).toFixed(1)}%;height:100%;border-radius:6px;
                    background:${i ? '#45443f'
                      : `linear-gradient(90deg,#2a78d6,${BLUE})`};
                    ${i ? '' : `box-shadow:0 0 14px ${BLUE}88`}"></div>
      </div>
      <div style="width:48px;text-align:right;font-variant-numeric:tabular-nums;
                  font:700 13.5px ui-monospace,Menlo,monospace;
                  color:${i ? DIM : '#fff'}">${p.toFixed(2)}</div>
    </div>`;

  // sparkline of every decision so far, so the *consistency* of the latency
  // is visible, not just the current number
  const peak = Math.max(...d.history, 1);
  const spark = d.history.map((ms, i) => `
    <div title="${ms} ms" style="width:7px;border-radius:2px;
        height:${Math.max(3, Math.round(ms / peak * 26))}px;
        background:${i === d.history.length - 1 ? ORANGE : '#3a3935'}"></div>`).join('');

  const crumbs = d.path.map((t, i) => `<span style="color:${
      i === d.path.length - 1 ? '#fff' : DIM}">${t}</span>`)
    .join(`<span style="color:#45443f"> &rsaquo; </span>`);

  const done = d.arrived;

  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="width:9px;height:9px;border-radius:50%;
                  background:${done ? '#1baf7a' : ORANGE};
                  box-shadow:0 0 12px ${done ? '#1baf7a' : ORANGE}"></div>
      <div style="font:750 14.5px -apple-system,system-ui,sans-serif;color:#fff;
                  letter-spacing:-.01em">open-bonzi-jev</div>
      <div style="font:600 12px ui-monospace,Menlo,monospace;color:${BLUE}">
        ${d.model} &middot; ${d.size} GB</div>
      <div style="margin-left:auto;font:600 10.5px -apple-system,system-ui,sans-serif;
                  color:${DIM};text-transform:uppercase;letter-spacing:.09em">
        1 forward pass per click</div>
    </div>

    <div style="display:flex;align-items:center;gap:9px;margin-bottom:11px;
                font:650 14px -apple-system,system-ui,sans-serif">
      <span style="color:#b8b7ad">${d.start}</span>
      <span style="color:#45443f">&rarr;</span>
      <span style="color:${ORANGE}">${d.goal}</span>
      ${done ? `<span style="margin-left:10px;padding:3px 9px;border-radius:99px;
          background:#1baf7a22;border:1px solid #1baf7a66;color:#1baf7a;
          font:700 11px -apple-system,system-ui,sans-serif;letter-spacing:.04em">
          REACHED IN ${d.step} CLICKS</span>` : ''}
      <span style="margin-left:auto;font:600 11px ui-monospace,Menlo,monospace;color:${DIM}">
        click ${d.step} &middot; ${d.options} options</span>
    </div>

    ${done ? '' : d.top.map(row).join('')}

    <div style="display:flex;align-items:flex-end;gap:14px;margin-top:14px;
                padding-top:12px;border-top:1px solid ${LINE}">
      <div>
        <div style="font:750 34px ui-monospace,Menlo,monospace;color:#fff;
                    font-variant-numeric:tabular-nums;line-height:.95;
                    letter-spacing:-.02em">${d.ms}<span
           style="font:600 14px -apple-system,system-ui,sans-serif;color:${DIM};
                  margin-left:5px">ms</span></div>
        <div style="font:500 10.5px -apple-system,system-ui,sans-serif;color:${DIM};
                    text-transform:uppercase;letter-spacing:.09em;margin-top:3px">
          to decide</div>
      </div>
      <div style="display:flex;align-items:flex-end;gap:3px;height:28px;margin-left:4px">
        ${spark}</div>
      <div style="margin-left:auto;text-align:right">
        <div style="font:700 15px ui-monospace,Menlo,monospace;color:#fff;
                    font-variant-numeric:tabular-nums">${(d.total_ms/1000).toFixed(1)} s</div>
        <div style="font:500 10.5px -apple-system,system-ui,sans-serif;color:${DIM};
                    text-transform:uppercase;letter-spacing:.09em;margin-top:3px">
          model time total</div>
      </div>
    </div>

    <div style="margin-top:10px;font:400 11px -apple-system,system-ui,sans-serif;
                line-height:1.45;color:${DIM}">${crumbs}</div>`;

  Object.assign(el.style, {
    position: 'fixed', left: '24px', bottom: '24px', width: '680px',
    padding: '17px 19px', color: '#fff', zIndex: 2147483647,
    background: 'linear-gradient(180deg,rgba(20,20,19,.975),rgba(11,11,10,.975))',
    borderRadius: '16px', border: '1px solid #2f2e2a',
    boxShadow: '0 20px 60px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.05)',
    backdropFilter: 'blur(8px)',
    font: '400 13px -apple-system,system-ui,sans-serif',
  });
  document.body.appendChild(el);
}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="the Wikipedia game, by typed decision")
    ap.add_argument("--start", default="LangChain")
    ap.add_argument("--goal", default="Microphone")
    ap.add_argument("--model", default="Bonsai-8B")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--settle-ms", type=int, default=250)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--dwell", type=int, default=900,
                    help="ms the HUD stays up per click, for recording")
    ap.add_argument("--expected-clicks", type=int, default=8,
                    help="only drives the HUD progress bar")
    ap.add_argument("--json")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
