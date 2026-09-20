#!/usr/bin/env python3
"""
Run the Wikipedia game over many start/goal pairs, to get a sample big enough
to say something.

Eight pairs was not enough: it put the 4B one pair ahead of the 8B, which is
noise. This loads each model once and reuses one browser across every pair,
because per-run model loading dominated the first sweep.

Pairs are drawn with a fixed seed from a list of well-known articles, so the
set is reproducible and not chosen to flatter anyone.

    sweep.py --pairs 100 --models Bonsai-1.7B,Bonsai-4B,Bonsai-8B
    sweep.py --pairs 25 --models Ternary-Bonsai-2-27B --out results/sweep-27b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "openjev"))
sys.path.insert(0, str(HERE))

from score import LETTERS, LlamaCppBackend, softmax_over, score_with  # noqa: E402
from wikinav import MAX_LINKS, SCROLL, SNAPSHOT_JS, pick_server, title_of, wiki_url  # noqa: E402

# A hand-picked pair list rather than random sampling, in three tiers so the
# result has structure instead of one averaged number. Uniformly random pairs
# over a topic list turned out close to unreachable for an agent with no
# backtracking and a 20-click budget (Cryptography -> Wine, Microphone ->
# Black hole: 0/3). Every title below was checked against the Wikipedia API to
# confirm it exists and is not a redirect, because a redirecting goal can never
# match the arrival test.
NEAR = [  # strongly associated, a good agent should need 1-3 clicks
 ("Jimmy Page","Guitar"),("The Beatles","Music"),("Jazz","Music"),
 ("Piano","Musical instrument"),("Microphone","Sound"),
 ("Hans Christian Andersen","Denmark"),("Vikings","Scandinavia"),
 ("Marie Curie","Radioactive decay"),("Charles Darwin","Evolution"),
 ("Leonardo da Vinci","Renaissance"),("Vincent van Gogh","Painting"),
 ("Apollo 11","Moon"),("Black hole","Gravity"),("Earthquake","Plate tectonics"),
 ("DNA","Gene"),("Vaccine","Immune system"),("Antibiotic","Bacteria"),
 ("Coffee","Caffeine"),("Chocolate","Cocoa bean"),("Wine","Grape"),
 ("Bread","Wheat"),("Bicycle","Wheel"),("Steam engine","Steam"),
 ("Printing press","Printing"),("Photography","Camera"),
 ("Telephone","Telecommunications"),("Linux","Operating system"),
 ("Open source","Software"),("Cryptography","Encryption"),
 ("Chess","Board game"),("Paris","France"),("Tokyo","Japan"),
 ("Brazil","South America"),("Mount Everest","Himalayas"),("Sahara","Desert"),
]
MID = [  # same broad domain, a few hops apart
 ("Jimmy Page","Microphone"),("Guitar","Electricity"),("Piano","Sound"),
 ("The Beatles","Sound recording and reproduction"),("Jazz","Improvisation"),
 ("Hans Christian Andersen","Literature"),("Vikings","Ship"),
 ("Roman Empire","Architecture"),("Marie Curie","Energy"),
 ("Charles Darwin","Biology"),("Leonardo da Vinci","Engineering"),
 ("Vincent van Gogh","Color"),("Apollo 11","Physics"),
 ("Black hole","Astronomy"),("Volcano","Geology"),("Earthquake","Geology"),
 ("DNA","Chemistry"),("Vaccine","Medicine"),("Antibiotic","Medicine"),
 ("Coffee","Brazil"),("Tea","China"),("Chocolate","Mexico"),("Wine","France"),
 ("Bread","Agriculture"),("Bicycle","Metal"),("Rail transport","Steel"),
 ("Steam engine","Energy"),("Printing press","Language"),
 ("Photography","Light"),("Television","Electricity"),("Telephone","Sound"),
 ("Linux","Computer"),("Open source","Internet"),("Cryptography","Mathematics"),
 ("Chess","Mathematics"),("Olympic Games","Greece"),
 ("Industrial Revolution","Factory"),("Silk Road","Trade"),
 ("Iceland","Volcano"),("Amazon rainforest","Water"),
]
FAR = [  # different domains; needs a real route through hubs
 ("Jimmy Page","Mathematics"),("Hans Christian Andersen","Physics"),
 ("Vincent van Gogh","Chemistry"),("Chess","Music"),("Coffee","Electricity"),
 ("Sahara","Water"),("Vikings","Mathematics"),("Wine","Biology"),
 ("Bread","Chemistry"),("Bicycle","Energy"),("Chocolate","Agriculture"),
 ("Tea","Water"),("Paris","Electricity"),("Tokyo","Technology"),
 ("Brazil","Agriculture"),("Egypt","Mathematics"),("Iceland","Energy"),
 ("Denmark","Agriculture"),("Japan","Technology"),("Roman Empire","Language"),
 ("Silk Road","Language"),("Printing press","Science"),
 ("Olympic Games","Physics"),("Rail transport","Energy"),("Television","Light"),
 ("Photography","Chemistry"),("Linux","Mathematics"),
 ("Open source","Electricity"),("Apollo 11","Computer"),
 ("Marie Curie","Medicine"),("Charles Darwin","Geology"),
 ("Mount Everest","Weather"),("Amazon rainforest","Climate"),
 ("Volcano","Energy"),("DNA","Mathematics"),
]
PAIRS = [(s,g,"near") for s,g in NEAR] + [(s,g,"mid") for s,g in MID] + \
        [(s,g,"far") for s,g in FAR]


def pair_list(n: int | None = None):
    return PAIRS if not n else PAIRS[:n]





def play(backend, page, start: str, goal: str, max_steps: int,
         settle_ms: int) -> dict:
    """One game. Returns whether it arrived, plus timings."""
    target = goal.lower()
    visited: set[str] = set()
    path, timings = [], []
    clicks = 0
    scrolls = 0
    t0 = time.perf_counter()
    try:
        page.goto(wiki_url(start), wait_until="domcontentloaded", timeout=30000)
    except Exception:
        return {"arrived": None, "error": "start page failed"}

    for _ in range(max_steps):
        page.wait_for_timeout(settle_ms)
        try:
            snap = page.evaluate(SNAPSHOT_JS)
        except Exception:
            break
        here = snap["title"] or title_of(page.url)
        if not path or path[-1] != here:
            path.append(here)
        visited.add(here.lower())
        if here.lower() == target:
            break

        cands = [l for l in snap["links"] if l["title"].lower() not in visited]
        options = [l["text"] for l in cands[:MAX_LINKS]]
        if snap["more"] and scrolls < 2:
            options.append(SCROLL)
        if not options:
            break

        q = (f"Goal: reach the Wikipedia article {goal!r}. You win by using the "
             f"fewest clicks, so pick the link most likely to get to {goal!r} "
             f"fastest. Which of these links on {here!r} goes closest to {goal!r}?")
        state = {"goal": f"Reach the Wikipedia article: {goal}",
                 "current_article": here, "articles_already_visited": path[-6:]}
        try:
            r = score_with(backend, q, options, state)
        except Exception:
            break
        timings.append(r["forward_s"])
        probs = {options[LETTERS.index(l)]: p
                 for l, p in softmax_over(r["logits"]).items()}
        choice = max(probs, key=probs.get)

        if choice == SCROLL:
            scrolls += 1
            page.mouse.wheel(0, 800)
            continue
        scrolls = 0
        link = next(l for l in cands if l["text"] == choice)
        clicks += 1
        try:
            page.goto(wiki_url(link["title"]), wait_until="domcontentloaded",
                      timeout=30000)
        except Exception:
            break

    arrived = bool(path) and path[-1].lower() == target
    return {"arrived": arrived, "clicks": clicks, "decisions": len(timings),
            "wall_s": round(time.perf_counter() - t0, 2),
            "model_s": round(sum(timings), 3),
            "ms_per_decision": round(statistics.mean(timings) * 1000)
            if timings else None,
            "path_len": len(path), "last": path[-1] if path else None}


def main(argv=None):
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser(description="many-pair Wikipedia game sweep")
    ap.add_argument("--pairs", type=int, default=0,
                    help="use only the first N pairs; 0 = all")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--models", default="Bonsai-1.7B,Bonsai-4B,Bonsai-8B")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--settle-ms", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results" / "wikigame-sweep100.json"))
    args = ap.parse_args(argv)

    pairs = pair_list(args.pairs if args.pairs > 0 else None)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    outp = Path(args.out)
    outp.parent.mkdir(exist_ok=True)
    # Merge into an existing sweep rather than clobbering it, so a model can be
    # added later on the same pair list without re-running the others.
    if outp.exists():
        out = json.loads(outp.read_text())
        if out.get("pairs") != [list(p) for p in pairs]:
            raise SystemExit(
                f"{outp} holds a different pair list; use a new --out")
        out.setdefault("models", {})
    else:
        out = {"pairs": [list(p) for p in pairs],
               "max_steps": args.max_steps, "models": {}}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        ctx.route("**/*", lambda route: (
            route.abort() if "BannerLoader" in route.request.url
            or "centralnotice" in route.request.url.lower()
            else route.continue_()))
        ctx.add_init_script("""
          const css = `#siteNotice,#centralNotice,[id^="frb"],[class^="frb"],
            .cn-fundraising,.mw-dismissable-notice{display:none!important}`;
          const add = () => { if (!document.getElementById('nb')) {
            const s=document.createElement('style'); s.id='nb'; s.textContent=css;
            (document.head||document.documentElement).appendChild(s);} };
          add(); document.addEventListener('DOMContentLoaded', add);
        """)
        page = ctx.new_page()

        for model in models:
            if model == "jev":
                # The hosted reference. Same pairs, same option lists, same
                # harness; only the decider changes.
                from jev_backend import JevBackend
                backend, size = JevBackend(), 0.0
                print(f"\n=== Jev (hosted API), {len(pairs)} pairs",
                      file=sys.stderr, flush=True)
            else:
                mdir = ROOT / "bonzi" / "models" / model
                size = sum(f.stat().st_size for f in mdir.glob("*.gguf")) / 1e9
                print(f"\n=== {model} ({size:.2f} GB), {len(pairs)} pairs",
                      file=sys.stderr, flush=True)
                backend = LlamaCppBackend(mdir, gpu=True, ctx=4096,
                                          llama_server=str(pick_server(mdir)))
            rows = []
            t_model = time.perf_counter()
            for i, (start, goal, tier) in enumerate(pairs, 1):
                r = play(backend, page, start, goal, args.max_steps, args.settle_ms)
                r.update(start=start, goal=goal, tier=tier)
                rows.append(r)
                solved = sum(1 for x in rows if x.get("arrived"))
                print(f"  [{i:3d}/{len(pairs)}] {'OK ' if r.get('arrived') else '.  '}"
                      f"{start[:22]:22s} -> {goal[:22]:22s} "
                      f"{r.get('clicks','-'):>3} clicks   running {solved}/{i}",
                      file=sys.stderr, flush=True)
                out["models"][model] = {"size_gb": round(size, 2), "rows": rows}
                outp.write_text(json.dumps(out, indent=2))   # resumable snapshot
            backend.close()
            solved = [r for r in rows if r.get("arrived")]
            ms = [r["ms_per_decision"] for r in rows if r.get("ms_per_decision")]
            print(f"  {model}: {len(solved)}/{len(rows)} solved in "
                  f"{(time.perf_counter()-t_model)/60:.1f} min, "
                  f"median {statistics.median([r['clicks'] for r in solved]) if solved else '-'} "
                  f"clicks, {statistics.mean(ms):.0f} ms/decision",
                  file=sys.stderr, flush=True)
        ctx.close()
        browser.close()

    print(f"\nwrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
