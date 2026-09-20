#!/usr/bin/env python3
"""
Find things on Wikipedia with a small Bonsai model: one forward pass per answer.

The task a person actually does: type a question into Wikipedia, look at the
results, and click the right one. The looking-and-clicking is the decision, and
it is exactly one forward pass here. Nothing is generated and nothing is parsed.

    Q: "the Danish author who wrote The Little Mermaid"
    A. Hans Christian Andersen       B. The Little Mermaid
    C. The Little Mermaid (1989 film) D. Danish literature   ...

This is the [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
shape: an indexed action space over what is on screen, one typed decision to
pick the target. jev-ultrafast calls a small LLM to write text; we skip that by
using the query itself as the search string, so the model does nothing but
decide.

    wikifind.py                                   # the built-in question set
    wikifind.py --model Bonsai-1.7B --video
    wikifind.py -q "the physicist who developed general relativity"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "openjev"))

from score import LETTERS, LlamaCppBackend, build_messages, softmax_over  # noqa: E402

MAX_OPTIONS = 24

# (natural-language question, the article that answers it).
#
# Every question here was checked so that the correct article really is in
# Wikipedia's top 12 fulltext results. That matters: of an earlier set, three
# answers (Mount Everest, Tokyo, Nile) never appeared in the results at all, so
# no model could have picked them, and scoring them as model errors would have
# been scoring Wikipedia's search engine instead. The answer's rank in the
# result list is noted, because rank 1 is not a real decision.
QUESTIONS = [
    ("the Danish author who wrote The Little Mermaid and The Ugly Duckling",
     "Hans Christian Andersen"),                                    # rank 1
    ("the ship that sank after hitting an iceberg in 1912", "Titanic"),        # rank 2
    ("the war fought in the United States between 1861 and 1865",
     "American Civil War"),                                          # rank 2
    ("the ocean between Africa and Australia", "Indian Ocean"),      # rank 6
    ("the spacecraft mission that first landed humans on the Moon", "Apollo 11"),  # rank 6
    ("the physicist who developed the theory of general relativity",
     "Albert Einstein"),                                             # rank 9
    ("the Italian artist who painted the Mona Lisa", "Leonardo da Vinci"),     # rank 10
    ("the high-level programming language created by Guido van Rossum",
     "Python (programming language)"),                               # rank 1
    ("the Danish physicist who founded quantum atomic theory", "Niels Bohr"),  # rank 1
    ("the 1986 nuclear accident in Soviet Ukraine", "Chernobyl disaster"),     # rank 1
]

SEARCH = ("https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search=")

RESULTS_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  for (const li of document.querySelectorAll('.mw-search-result-heading a, li.mw-search-result a')) {
    const raw = li.getAttribute('href') || '';
    const i = raw.indexOf('/wiki/');
    if (i === -1) continue;
    const title = decodeURIComponent(raw.slice(i + 6).split('#')[0]).replace(/_/g, ' ');
    const key = title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const row = li.closest('li');
    const snip = row ? (row.querySelector('.searchresult') || {}).textContent || '' : '';
    out.push({ title, snippet: snip.trim().replace(/\s+/g, ' ').slice(0, 160) });
  }
  return out;
}
"""


# An overlay drawn on the destination page so a screen recording shows the
# decision, not just the result: the question, what the model weighed, and how
# long the forward pass took.
HUD_JS = r"""
(d) => {
  document.getElementById('bonzi-hud')?.remove();
  const el = document.createElement('div');
  el.id = 'bonzi-hud';
  const bars = d.top.map(([name, p], i) => `
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
      <div style="width:250px;white-space:nowrap;overflow:hidden;
                  text-overflow:ellipsis;color:${i ? '#c3c2b7' : '#fff'};
                  font-weight:${i ? 400 : 600}">${name}</div>
      <div style="flex:1;height:8px;background:#33322f;border-radius:4px;overflow:hidden">
        <div style="width:${(p * 100).toFixed(1)}%;height:100%;
                    background:${i ? '#52514e' : '#2a78d6'};border-radius:4px"></div>
      </div>
      <div style="width:44px;text-align:right;font-variant-numeric:tabular-nums;
                  color:${i ? '#8a897f' : '#fff'}">${p.toFixed(2)}</div>
    </div>`).join('');
  el.innerHTML = `
    <div style="font:600 13px -apple-system,system-ui,sans-serif;color:#fff;
                margin-bottom:2px">${d.model} &middot; ${d.size} GB &middot; one forward pass</div>
    <div style="font:400 12px -apple-system,system-ui,sans-serif;color:#c3c2b7;
                margin-bottom:8px">Q ${d.q}</div>
    <div style="font:400 12px ui-monospace,Menlo,monospace">${bars}</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px;
                font:600 12px ui-monospace,Menlo,monospace;color:#3987e5">
      <span>${d.ms} ms to decide</span>
      <span style="color:#8a897f">${d.i}/${d.n} &middot; ${d.options} options</span>
    </div>`;
  Object.assign(el.style, {
    position: 'fixed', left: '24px', bottom: '24px', width: '620px',
    padding: '14px 16px', background: 'rgba(16,16,15,0.95)', color: '#fff',
    borderRadius: '10px', zIndex: 2147483647,
    boxShadow: '0 8px 32px rgba(0,0,0,0.45)',
    font: '400 12px -apple-system,system-ui,sans-serif',
  });
  document.body.appendChild(el);
}
"""


def find_one(backend, page, question, timings, max_options=MAX_OPTIONS,
             snippet_chars=0) -> dict:
    """Search, then one forward pass to pick the article that answers it."""
    t_nav = time.perf_counter()
    page.goto(SEARCH + quote(question), wait_until="domcontentloaded")
    page.wait_for_timeout(200)
    results = page.evaluate(RESULTS_JS)
    nav_s = time.perf_counter() - t_nav
    if not results:
        return {"choice": None, "nav_s": nav_s, "forward_s": 0.0, "n": 0}

    results = results[:max_options]
    options = [r["title"] for r in results]
    # Snippets cost a lot of prompt: 24 results at 160 chars is ~4k tokens and
    # pushed one decision from 245 ms to 4.1 s on Bonsai 8B. Titles alone are
    # usually enough, because the option list already *is* the evidence.
    state = {"search_query": question}
    if snippet_chars:
        state["results"] = [{"title": r["title"],
                             "snippet": r["snippet"][:snippet_chars]} for r in results]
    r = backend.score(
        build_messages(
            f"Which of these Wikipedia articles is about {question}?",
            options, state),
        options)
    timings.append(r["forward_s"])
    probs = {options[LETTERS.index(l)]: p for l, p in softmax_over(r["logits"]).items()}
    choice = max(probs, key=probs.get)

    t_open = time.perf_counter()
    page.goto("https://en.wikipedia.org/wiki/" + choice.replace(" ", "_"),
              wait_until="domcontentloaded")
    open_s = time.perf_counter() - t_open
    # Grade on where we actually landed. Many correct answers are redirects:
    # "1986 Chernobyl Nuclear Accident" resolves to "Chernobyl disaster", and
    # scoring that wrong would be scoring Wikipedia's synonyms, not the model.
    landed = page.evaluate(
        "() => (document.querySelector('#firstHeading')||{}).textContent || ''"
    ).strip()
    return {"choice": choice, "landed": landed or choice,
            "p": probs[choice], "n": len(options),
            "top": sorted(probs.items(), key=lambda kv: -kv[1])[:3],
            "forward_s": r["forward_s"], "nav_s": nav_s, "open_s": open_s,
            "runner_up": sorted(probs.items(), key=lambda kv: -kv[1])[1:3]}



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

    questions = ([(args.question, None)] if args.question else QUESTIONS)[: args.limit]

    print(f"loading {args.model} ...", file=sys.stderr)
    t = time.perf_counter()
    backend = LlamaCppBackend(ROOT / "bonzi" / "models" / args.model, gpu=True,
                              ctx=4096, llama_server=str(server))
    load_s = time.perf_counter() - t
    size = sum(f.stat().st_size for f in
               (ROOT / "bonzi" / "models" / args.model).glob("*.gguf")) / 1e9
    print(f"{args.model} ({size:.2f} GB) loaded in {load_s:.1f}s\n", file=sys.stderr)

    timings, rows = [], []
    t0 = time.perf_counter()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        kw = {"viewport": {"width": 1280, "height": 800}}
        if args.video:
            (HERE / "video").mkdir(exist_ok=True)
            kw["record_video_dir"] = str(HERE / "video")
            kw["record_video_size"] = {"width": 1280, "height": 800}
        ctx = browser.new_context(**kw)
        page = ctx.new_page()

        for q, expected in questions:
            r = find_one(backend, page, q, timings,
                         args.max_options, args.snippet_chars)
            ok = expected is None or expected.lower() in {
                (r.get("choice") or "").lower(), (r.get("landed") or "").lower()}
            rows.append({"question": q, "expected": expected, **r, "correct": ok})
            mark = " " if expected is None else ("ok " if ok else "MISS")
            print(f"  {mark} {q[:52]:52s}")
            print(f"       -> {str(r['choice'])[:44]:44s} "
                  f"p={r.get('p', 0):.2f}  {r['forward_s']*1000:4.0f} ms "
                  f"({r['n']} options)")
            if r.get("landed") and r["landed"] != r["choice"]:
                print(f"          (redirected to {r['landed']!r})")
            if not ok and expected:
                print(f"          expected {expected!r}")
            if args.hud and r.get("top"):
                try:
                    page.evaluate(HUD_JS, {
                        "model": args.model, "size": f"{size:.2f}",
                        "q": q, "ms": round(r["forward_s"] * 1000),
                        "top": [[t, float(pv)] for t, pv in r["top"]],
                        "i": len(rows), "n": len(questions), "options": r["n"],
                    })
                except Exception:
                    pass
            if args.dwell:
                page.wait_for_timeout(args.dwell)

        total = time.perf_counter() - t0
        vpath = page.video.path() if args.video else None
        ctx.close()
        browser.close()
    backend.close()

    n = len(timings)
    graded = [r for r in rows if r["expected"]]
    print()
    if graded:
        print(f"  correct          {sum(r['correct'] for r in graded)}/{len(graded)}")
    if n:
        print(f"  decision latency {statistics.median(timings)*1000:.0f} ms median, "
              f"{min(timings)*1000:.0f}-{max(timings)*1000:.0f} ms range")
        print(f"  model time       {sum(timings):.2f} s of {total:.1f} s wall "
              f"({sum(timings)/total*100:.0f}%)")
    print(f"  wall clock       {total:.1f} s for {len(rows)} lookups "
          f"({total/len(rows):.1f} s each, page loads included)")
    if vpath:
        print(f"  video            {vpath}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": args.model, "size_gb": round(size, 2),
            "load_s": round(load_s, 2), "wall_s": round(total, 2),
            "model_s": round(sum(timings), 2),
            "median_ms": round(statistics.median(timings) * 1000) if n else None,
            "correct": sum(r["correct"] for r in graded), "graded": len(graded),
            "rows": rows,
        }, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="find articles on Wikipedia by typed decision")
    ap.add_argument("-q", "--question", help="one ad-hoc question")
    ap.add_argument("--model", default="Bonsai-8B")
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--dwell", type=int, default=0,
                    help="ms to linger on each article, for video legibility")
    ap.add_argument("--hud", action="store_true",
                    help="draw the decision on the page, for screen recording")
    ap.add_argument("--max-options", type=int, default=12,
                    help="search results offered per question (<=24)")
    ap.add_argument("--snippet-chars", type=int, default=0,
                    help="include this many chars of each result snippet; 0 = "
                         "titles only, which is far faster")
    ap.add_argument("--json")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
