# Browser use with a small Bonsai model

**A 1.16 GB model, 4 clicks, 5.4 seconds.**

<img src="wiki-game-8b.gif" alt="Bonsai 1 8B playing the Wikipedia game: Jimmy Page to Microphone in 4 clicks" width="100%">

**A harder start, where the small models drift and the 5.95 GB one does not.**

<img src="wiki-game-27b.gif" alt="Bonsai 2 27B playing the Wikipedia game: Open source to Microphone in 6 clicks" width="100%">

The Wikipedia game, played by a local model: reach one article from another by
clicking links only, no search. **Every click is one forward pass.** Nothing is
generated and nothing is parsed; the option letter with the highest probability
*is* the click.

```
A. Integrated circuit    B. Free software      C. Open-source hardware
D. Linux                 E. SCROLL_DOWN
```

## How the GIF was made

That recording is a real run, start to finish, at real speed. Six steps:

**1. Snapshot the page into an action space.** A script in the page collects
every article link that is on screen (plus one screen of lookahead), throws out
namespace pages, disambiguation pages and anything already visited, and caps
what is left at 24 so each option fits in a single letter token. `SCROLL_DOWN`
is appended only if there is more page and we have not just scrolled twice.

**2. Ask one question.** State is the goal, the current article and the pages
already visited; the options are that link list. The question leads with the
goal, never with the current page.

**3. One forward pass.** `openjev/score.py` renders the chat template with
thinking disabled, runs a single forward pass, reads the score of each option
letter and softmaxes over just those. No text is generated and nothing is
parsed.

**4. Click the argmax.** The highest-probability letter is the link to follow.
Repeat from step 1.

**5. Record, without an overlay.** Playwright records the 1280x800 viewport
only, and the run writes a timeline as it goes: one entry per arrival and per
decision, timestamped against the start of the recording.

**6. Composite the dashboard afterwards.** `make_video.py` renders one panel
image per timeline entry and has ffmpeg lay them over the video on a dark
canvas. Drawing the panel *after* the fact rather than injecting it into the
page keeps the measurement honest: an overlay cannot slow the agent down or
change which links were on screen.

```bash
# the run, recorded
PYTHONPATH=../../openjev ../../.venv-browser/bin/python wikinav.py \
    --start "Open source" --goal "Microphone" \
    --model Ternary-Bonsai-2-27B --video --json run.json

# the dashboard, composited on top
python3 make_video.py run.json --gif
```

Two things the recording needed. Wikimedia's fundraising banner is blocked at
the network layer and hidden with CSS, because it covers the article; note that
removing it **changes the run**, since it changes which links are on screen and
the screen is the action space. And the browser chrome in the GIF is the real
Chromium viewport, inset on the canvas by the compositor, not a mockup.

## Where this comes from

- **[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)** established the shape: snapshot the page into an *indexed* table of interactive elements, one request picks the operation and the target, a small LLM writes text only when typing is needed.
- **[ndrezn/ts-browser-agent](https://github.com/ndrezn/ts-browser-agent)** (LangChain + TypeSafe) built the wiki-game demo this reproduces, including its start/goal pairs.

Here the only operation is "follow a link", so no text model is involved at
all; the run is pure typed decisions.

## Results, measured

Apple M4 Pro, GPU, links only, no search. Runs are deterministic: argmax at
temperature 0, so a given model and start always walk the same path.

### Across 110 curated pairs

<img src="../../docs/charts/wikigame-solve-rate.svg" alt="Wikipedia game solve rate by difficulty tier, four Bonsai sizes and hosted Jev" width="100%">

110 hand-picked start→goal pairs in three tiers, 20-click cap, links only, one
deterministic run each. Every title was checked against the Wikipedia API so
none is a redirect, because a redirecting goal can never match the arrival
test. **Jev is included as the reference**: same pairs, same option lists, same
harness, only the decider changes.

| Model | Where | Overall | near (35) | mid (40) | far (35) | Median clicks | Per decision | Per pair |
|---|---|---|---|---|---|---|---|---|
| Bonsai 1 1.7B | 0.25 GB local | 58/110 (53%) | 32/35 | 20/40 | 6/35 | 3 | **139 ms** | 9.9 s |
| Bonsai 1 4B | 0.57 GB local | 91/110 (83%) | 31/35 | 34/40 | 26/35 | 4 | 346 ms | 9.1 s |
| Bonsai 1 8B | 1.16 GB local | 87/110 (79%) | 35/35 | 33/40 | 19/35 | 4 | 614 ms | 12.5 s |
| Bonsai 2 27B | 5.95 GB local | 106/110 (96%) | 35/35 | 38/40 | 33/35 | 3 | 4160 ms | 22.0 s |
| **Jev** | hosted API | **110/110 (100%)** | 35/35 | 40/40 | 35/35 | 4 | 818 ms | **5.9 s** |

Tiers are *near* (strongly associated, 1–3 clicks for a good agent), *mid*
(same broad domain), *far* (cross-domain, needs a real route through hub
pages). Pair list and raw results: `../../results/wikigame-sweep110.json`.

### Jev solved every pair. What that does and does not tell you

Exact McNemar tests on the paired per-pair outcomes:

| Comparison | Discordant | p | Verdict |
|---|---|---|---|
| 1.7B vs 4B | 5 / 38 | <0.0001 | **real** |
| **4B vs 8B** | 16 / 12 | 0.57 | **a tie** |
| 8B vs 27B | 0 / 19 | <0.0001 | **real** |
| 1.7B vs Jev | 0 / 52 | <0.0001 | **real** |
| 8B vs Jev | 0 / 23 | <0.0001 | **real** |
| **Bonsai 2 27B vs Jev** | **0 / 4** | **0.125** | **cannot tell** |

**Jev is the strongest thing here and it is not close for the small models.**
It is also notable that it never lost a pair to anything: every discordance in
the table above runs one way.

**Bonsai 2 27B versus Jev is 106 against 110, and the honest answer is that
this sample cannot separate them.** Be careful with that sentence, though. Four
discordant pairs all favour Jev, and with four discordant pairs the smallest
p an exact test can return is 0.125, so it *could not* have reached
significance whichever way they fell. This is an underpowered comparison, not
evidence of equivalence. Directionally Jev is ahead; more pairs would be needed
to measure by how much.

**Jev is also the fastest end to end**, at 5.9 s a pair, beating even the
0.25 GB model's 9.9 s despite an 818 ms network round trip per decision. Good
routes mean fewer clicks, and clicks mean page loads, which dominate. The 27B
takes 22.0 s a pair for nearly the same solve rate.

### The 27B's four failures were all one hop short

| Pair | Stopped at |
|---|---|
| Jazz → Improvisation | `Solo (music)` |
| Vaccine → Medicine | `Disease` |
| Paris → Electricity | `Electrical energy` |
| Tokyo → Technology | `History of communication` |

Every one landed on a semantically adjacent article and ran out of clicks
rather than wandering off. The remaining errors are the arrival test being
exact-title, not the navigation failing.

### Slower per decision, barely slower per pair

The 27B is **30× slower per decision** than the 1.7B (4160 ms vs 139 ms) but
only **2.2× slower per pair** (22.0 s vs 9.9 s), because it needs fewer clicks
and, for the small models, page loads dominate. The 4B is the throughput
winner outright: highest solve rate per second of wall clock, at 9.1 s a pair.

### Individual runs

| Model | On disk | Jimmy Page → Microphone | Open source → Microphone |
|---|---|---|---|
| Bonsai 1 1.7B | 0.25 GB | 15 clicks, 13.2 s | failed at 22 |
| Bonsai 1 4B | 0.57 GB | **4 clicks, 4.0 s** | 13 clicks, 13.2 s |
| Bonsai 1 8B | 1.16 GB | 4 clicks, 5.2 s | failed at 22 |
| Bonsai 2 27B | 5.95 GB |, | 6 clicks, 28.6 s |

### Why a bigger model can lose a pair: knowledge is not navigation

`Open source → Microphone` looked like a bug, so we replayed the exact decision
where the two models diverge. Both reach `Open-source hardware` and are handed
the **same 25 options**. Then:

| | Bonsai 1 4B | Bonsai 1 8B |
|---|---|---|
| top choice | `schematics` **0.74** | `RepRap` **0.97** |
| leads to | Circuit diagram → electrical wiring → conductor → … → Microphone | RepRap → self-replicating machines → nanotechnology → … nowhere |

Not a coin flip and not a bug. The 8B is *more confident and more on-topic*:
RepRap is the canonical open-source hardware project. It is simply not on the
way to a microphone. The 4B's `schematics` is a vaguer, more general link, and
general links tend to be hub pages with more outward routes.

That is the whole effect. The Wikipedia game rewards **picking hubs**, while
knowing a topic well pulls you toward its specific, well-associated leaves.
Neither model reasoned about a route to microphones; one was dragged toward a
dead end by knowing more.

This explains individual pairs, not the overall standings: across all 110
pairs the 4B and 8B are a statistical tie, and the 27B beats both. Knowledge
does not hurt on average, it just does not help the way it helps on WANLI.

### Other things worth knowing

**Filtering mattered more than model size.** Every number here improved when
disambiguation pages were removed from the action space. Before that change the
4B and 1.7B failed `Jimmy Page → Microphone` outright and the 8B took 8 clicks
instead of 4. A model that keeps landing on `Shrink wrap → Shrink Rap → a TV
series` looks stupid; it is mostly being offered stupid options.

**A better decision is still worth more than a faster one.** On the hard start
Bonsai 2 27B is 6× slower per decision than the 8B and finishes where the 8B
does not, in 6 clicks rather than wandering for 22.

**Starting from [Main Page](https://en.wikipedia.org/wiki/Main_Page) does not
work** and is not reproducible: its links are the day's featured articles, so
the run changes daily. The 8B drifted through ships into fantasy literature.

## Running it

```bash
../../scripts/build_llama_fork.sh          # only needed for Bonsai 2
python3 ../../scripts/download_model.py Ternary-Bonsai-2-27B

PYTHONPATH=../../openjev ../../.venv-browser/bin/python wikinav.py \
    --start "Open source" --goal "Microphone" --model Ternary-Bonsai-2-27B
```

Setup: `python3 -m venv .venv-browser && ./.venv-browser/bin/pip install
playwright && ./.venv-browser/bin/playwright install chromium`.

The runtime is chosen from the weights, not from whichever binary exists, so
pointing it at a Bonsai 2 model picks PrismML's build automatically.

### The video

`wikinav.py --video --json run.json` records the page and a timeline; the
dashboard is composited afterwards:

```bash
python3 make_video.py run.json --gif
```

The panel is drawn **after** the run rather than injected into the page, so the
recording stays honest: the overlay cannot slow the agent down or change what
the page looked like.

## Design notes

**Rules are enforced by removing options, never by instructing.** This is
ts-browser-agent's finding and it reproduced exactly:

- Offer `DONE` and Bonsai 8B picks it on step one at p=0.88, standing on the
  start page.
- Offer unbounded `SCROLL_DOWN` and it scrolls at p=0.99 forever, eight steps
  running, never committing. It is capped at two in a row.
- Tell it "do not go back" and it goes back. Visited pages are filtered out
  instead.
- Disambiguation pages are a reliable trap (`Shrink wrap → Shrink Rap → a TV
  series`), so they are filtered too.

**Put the goal first.** ts-browser-agent noted the classifier weighs the
opening of the question heavily. Rewriting ours from "You are on X and want to
reach Y" to "Goal: reach Y … which link on X" turned a failed LangChain run
into a 10-click success.

**The viewport defines the action space.** Links are taken from the current
screen plus one screen of lookahead, capped at 24 so every option is a single
letter token. Suppressing Wikipedia's fundraising banner measurably changed a
run's path, because it changes which links are on screen.

**Wikipedia serves Parsoid HTML.** Article links are absolute
(`href="https://en.wikipedia.org/wiki/Foo"`), not relative `/wiki/Foo`, and the
body is under `#bodyContent`, not the near-empty `#mw-content-text`. Getting
this wrong yields three links per page instead of sixty.

## `wikifind.py`: the easier, search-assisted variant

Also here, and worth being upfront about: it types the question into
Wikipedia's search and picks among the results. **That is a much easier task**,
because the search engine does the retrieval and the answer is often rank 1.

It is still useful for isolating decision quality at a fixed action space:

| Model | On disk | Correct | Per decision |
|---|---|---|---|
| Bonsai 1 1.7B | 0.25 GB | 5/10 | 103 ms |
| Bonsai 1 4B | 0.57 GB | 8/10 | 231 ms |
| Bonsai 1 8B | 1.16 GB | 9/10 | 407 ms |

Two findings from it:

- **Snippets hurt.** Feeding 120 chars of each result's snippet made the model
  both slower (2050 ms vs 409 ms, from a ~4k-token prompt) and *less* accurate
  (3/8 vs 5/8). Titles alone win.
- **Question sets have to be checked.** In an earlier set, three answers
  (Mount Everest, Tokyo, Nile) were not in Wikipedia's top 12 results at all,
  so no model could have picked them. Scoring those as model errors would have
  been scoring the search engine.
