<h1 align="center">open-bonzi-jev</h1>

<p align="center">
  <b>Typed decisions from a 27B model, read out of one forward pass, from a 5.95&nbsp;GB file on a laptop.</b><br>
  <sub>openjev's mechanism &middot; Bonsai's weights &middot; runs on CPU, Metal, CUDA or Vulkan</sub>
</p>

<p align="center">
  <a href="#results"><img alt="WANLI-256 74.6%" src="https://img.shields.io/badge/WANLI--256-74.6%25-2a78d6"></a>
  <a href="#results"><img alt="5.95 GB" src="https://img.shields.io/badge/weights-5.95_GB-eb6834"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/code-MIT-informational"></a>
  <a href="https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf"><img alt="Apache 2.0 weights" src="https://img.shields.io/badge/weights-Apache_2.0-informational"></a>
</p>

```console
$ ./openjev/score.sh -q "Is the user angry?" --state '{"msg":"this is broken AGAIN and nobody replies"}'
{"choice": "true", "probabilities": {"true": 0.991651, "false": 0.008349},
 "label_mass": 0.994205, "prompt_tokens": 112, "forward_s": 1.852, "backend": "llama.cpp"}
```

Nothing is generated. Nothing is parsed. The model is handed a lettered
multiple-choice question, and the scorer reads the probability of `A` versus
`B` straight off the next-token distribution. One forward pass, a real number
out, an `if` you can branch on.

---

## Where this came from

This stands on three pieces of other people's work.

**[TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**
built **Jev** and gave the category a name: "System One" models that return
typed probabilistic decisions instead of text. It is closed, hosted, and behind
a waitlist.

**[TheoLeeCJ](https://github.com/TheoLeeCJ)** then showed you don't need any of
that. **[SemIf](https://github.com/TheoLeeCJ/SemIf)**, first released under the
name **OpenJev**, gets the same interface out of ordinary frozen open models by
reading the option logits directly, no training involved. It's MIT, and it runs
in your browser at [openjev.com](https://openjev.com).

**[PrismML](https://prismml.com)** built **Bonsai**, the low-bit models that
make it cheap: a 27B in 5.95 GB, and a usable one in 0.25 GB. Apache 2.0, used
exactly as published here, with nothing retrained or requantized.

What this repo does is put the second on top of the third, and then measure it:
against hosted Jev on the same eval files, across a 110-pair browser-agent
benchmark, for calibration, and on images.

If you want the reference implementation of the idea itself, that is
**[SemIf](https://github.com/TheoLeeCJ/SemIf)**, not this. It has the browser
build, shared-prefix reuse, a reranker comparison, perturbation tests and a
more careful eval harness.

`openjev/score.py` is an **independent re-implementation** of SemIf's "direct"
mode with its own prompt, and no SemIf code is vendored. Since the prompt differs,
**the numbers here aren't directly comparable to theirs.** Full accounting in
[CREDITS.md](CREDITS.md).

> Not affiliated with, endorsed by, or connected to TypeSafe, PrismML, or
> TheoLeeCJ. Jev and TypeSafe are referenced only to say what kind of thing
> this is.

---

## Results

Apple M4 Pro, GPU, context 4096, warmup excluded, one loaded model per run.
Every number below was produced by [`scripts/benchmark.py`](scripts/benchmark.py)
in this repo. Re-run it and you should get the same thing.

### Decision quality

<img src="docs/charts/wanli256-accuracy.svg" alt="WANLI-256 accuracy by Bonsai model" width="100%">

**WANLI-256** is the real measurement: 256 rows of adversarial natural-language
inference from [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI),
label-balanced so chance is 33.3% rather than the majority-class rate.

| Model | Quant | On disk | Runtime | easy100 | **WANLI-256** | q/min | median latency |
|---|---|---|---|---|---|---|---|
| **Bonsai 1 1.7B** | `Q1_0` | 0.25 GB | stock | 79% | **52.0%** | 925 | 57 ms |
| **Bonsai 1 4B** | `Q1_0` | 0.57 GB | stock | 97% | **60.2%** | 398 | 140 ms |
| **Bonsai 1 8B** | `Q1_0` | 1.16 GB | stock | 98% | **64.5%** | 230 | 245 ms |
| **Bonsai 1 27B** | `Q1_0` | 3.80 GB | stock | 100% | **71.1%** | 31 | 1911 ms |
| **Ternary Bonsai 1 8B** | `PQ2_0` | 2.18 GB | fork | 100% | **65.2%** | 200 | 281 ms |
| **Bonsai 2 27B** | `PTQ1_0` | 5.95 GB | fork | 100% | **74.6%** | 24 | 2443 ms |
| **Jev** (hosted) | n/a | n/a | TypeSafe API | 100% | **79.7%** | 75 | 774 ms |

**Hosted Jev on the identical eval file scores 79.7%** (204/256), against
74.6% for Bonsai 2 27B running locally from 5.95 GB. An exact McNemar test on
the paired items puts that gap at p=0.066, so it does not clear the usual bar
either, though Jev is clearly directionally ahead.

For scale, SemIf reports **0.637 balanced accuracy on WANLI-256 with
Qwen3.5-4B** at 3.01 GB. Bonsai 1 8B lands at 64.5% from 1.16 GB and Bonsai 2
27B at 74.6% from 5.95 GB. But **this is indicative, not a like-for-like
comparison**: different rows, different prompt, different model. Treat it as a
sanity range, not a leaderboard.

### Calibration: the probabilities lie, and one number fixes it

<img src="docs/charts/reliability.svg" alt="Reliability diagram: claimed probability against observed accuracy" width="100%">

Accuracy is only half of what makes a decision usable. The other half is
whether the probability can be believed, and out of the box it cannot:

| Model | Accuracy | Mean confidence | Calibration error | When it claims ≥0.99 it is right |
|---|---|---|---|---|
| Bonsai 1 8B | 0.645 | 0.895 | 0.238 | **78%** |
| Bonsai 2 27B | 0.746 | 0.899 | 0.176 | **86%** |
| Jev | 0.797 | 0.883 | 0.104 | **96%** |

Bonsai 2 27B saying "certain" is wrong one time in seven. That is the same
defect as answering *"does a spider have two legs?"* with `true` at 0.998.

**The fix is one divide**, before the softmax the scorer already runs:

```python
p = softmax(logits / T)
```

| Model | T | Calibration error, raw → calibrated | Accuracy |
|---|---|---|---|
| Bonsai 1 1.7B | 3.70 | 0.172 → 0.098 | unchanged |
| Bonsai 1 8B | 3.85 | 0.238 → 0.106 | unchanged |
| Bonsai 1 27B | 1.85 | 0.100 → 0.057 | unchanged |
| **Bonsai 2 27B** | **2.45** | **0.176 → 0.088** | unchanged |
| Jev | 1.75 | 0.104 → 0.061 | unchanged |

Calibrated, **Bonsai 2 27B lands at 0.088, where Jev sits untuned (0.104)**.
The whole trust gap closes with one scalar, no training and no inference cost.
T is fitted on half the eval and scored on the other half.

Three things worth being precise about:

- **It cannot change an answer.** Dividing by a positive constant preserves the
  ordering of the options, so accuracy is identical. Verified across every
  model and row: **0 changes in 1,858**. It fixes trust, not correctness, and
  the 5-point accuracy gap to Jev survives it.
- **Jev is overconfident here too** (0.104 → 0.061 at T=1.75). Calibration is
  a property of a distribution, not a model, and theirs was presumably tuned on
  their own decision mix rather than WANLI.
- **No single constant works for everyone.** The 8B needs T≈3.85, Bonsai 1 27B
  T≈1.85. One number per model, fitted on ~100 labelled rows.

```bash
python3 scripts/calibrate.py                    # fit, write results/calibration.json
./openjev/score.sh -q "Is the sky blue?" --calibrated
```

### Size and speed

<img src="docs/charts/accuracy-vs-size.svg" alt="WANLI-256 accuracy against weights on disk, by model" width="100%">

<img src="docs/charts/throughput.svg" alt="Decisions per minute by model, log scale" width="100%">

The trade is stark and worth internalising: **Bonsai 1 1.7B is 38× faster than
Bonsai 2 27B and barely above chance.** Quality costs throughput, roughly
linearly in parameters, and the 1-bit family buys size, not speed.

### The `easy100` column is a smoke test, not a benchmark

It is 100 trivial true/false facts, and everything from 4B up saturates it. It
exists to catch a broken prompt or a broken backend. It is in the table only so
you can see it saturate.

It does show one thing worth knowing. Asked *"does a spider have two legs?"*:

| Model | Answer | Confidence | `label_mass` |
|---|---|---|---|
| Bonsai 1 8B | `true` ❌ | **0.998** | 0.9997 |
| Bonsai 1 1.7B | `true` ❌ | 0.754 | 0.9775 |
| Bonsai 2 27B | `false` ✅ | 0.996 | 0.9932 |

**Bonsai 1 8B is wrong at 0.998 confidence**, with a `label_mass` of 0.9997.
The model was unambiguously answering the question asked, and was unambiguously
wrong. This is the single most important caveat in this repo: **`label_mass`
tells you that you got an answer, never that it is right.** A confident wrong
number and a confident right one are the same shape.

---

## Browser use: the Wikipedia game

Reach one Wikipedia article from another by clicking links only, no search,
with **one forward pass per click**. Nothing is generated and nothing is
parsed: the option letter with the highest probability *is* the click.

**Bonsai 1 8B, 1.16 GB on disk. Jimmy Page to Microphone in 4 clicks, 5.4 seconds.**

<img src="demos/browser/wiki-game-8b.gif" alt="Bonsai 1 8B playing the Wikipedia game, Jimmy Page to Microphone in 4 clicks" width="100%">

**A harder start, where the small models drift and Bonsai 2 27B does not.**

<img src="demos/browser/wiki-game-27b.gif" alt="Bonsai 2 27B playing the Wikipedia game, Open source to Microphone in 6 clicks" width="100%">

### How it works

Each step snapshots the page into an indexed list of article links (on screen
plus one screen of lookahead, minus namespace pages, disambiguation pages and
anywhere already visited, capped at 24 so every option is a single letter
token), asks one question with the goal leading, runs **one forward pass**, and
clicks the argmax. The dashboard is composited afterwards from a recorded
timeline, never injected into the page, so it cannot slow the agent down or
change which links were on screen.

This reproduces the demo from
[ndrezn/ts-browser-agent](https://github.com/ndrezn/ts-browser-agent)
(LangChain + TypeSafe), which follows
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast).

Across **110 curated start→goal pairs** in three difficulty tiers, 20-click
cap, one deterministic run each, with **hosted Jev as the reference** on the
identical harness:

<img src="docs/charts/wikigame-solve-rate.svg" alt="Wikipedia game solve rate by difficulty tier, four Bonsai sizes and hosted Jev" width="100%">

| Model | Where | Overall | near | mid | far | Per decision | Per pair |
|---|---|---|---|---|---|---|---|
| Bonsai 1 1.7B | 0.25 GB local | 58/110 (53%) | 32/35 | 20/40 | 6/35 | **139 ms** | 9.9 s |
| Bonsai 1 4B | 0.57 GB local | 91/110 (83%) | 31/35 | 34/40 | 26/35 | 346 ms | 9.1 s |
| Bonsai 1 8B | 1.16 GB local | 87/110 (79%) | 35/35 | 33/40 | 19/35 | 614 ms | 12.5 s |
| Bonsai 2 27B | 5.95 GB local | 106/110 (96%) | 35/35 | 38/40 | 33/35 | 4160 ms | 22.0 s |
| **Jev** | hosted API | **110/110 (100%)** | 35/35 | 40/40 | 35/35 | 818 ms | **5.9 s** |

**Jev solved every pair**, and never lost one to any local model. Against
Bonsai 2 27B the gap is 110 vs 106 with only **4 discordant pairs**, and with
four discordances the smallest p an exact McNemar test can return is 0.125, so
this sample *cannot* separate them either way. That is an underpowered
comparison, not evidence of a tie; Jev is directionally ahead on both tasks.

Among the local models, scaling is two steps and a plateau: 1.7B→4B is real
(p<0.0001), **4B→8B is a tie** (p=0.57), 8B→27B is real (p<0.0001). Unlike
WANLI above, size does not predict navigation smoothly.

**Jev is also fastest end to end** at 5.9 s a pair, beating the 0.25 GB model's
9.9 s despite an 818 ms round trip, because better routes mean fewer page
loads, and page loads dominate.

Full write-up: **[demos/browser/](demos/browser/)**.

---

## Pictures

The same primitive works on images, because Bonsai 2 27B is a vision-language
model. Each photo is asked about itself, in one forward pass, with the answer
read off the next-token distribution.

<img src="demos/vision/vision-grid.jpg" alt="Six photos, each asked whether it shows what it shows, with the returned probability under each" width="100%">

```console
$ python3 demos/vision/vision_score.py -q "Is this a bike?" --image images/bicycle.jpg
{"choice": "true", "probabilities": {"true": 0.999, "false": 0.001}, "label_mass": 0.996}
```

**12/12 correct**, median 6.8 s per image. A six-way "what is the main subject?"
over the same photos is also perfect, every answer at p ≥ 0.998.

**This is the one thing the local model does that Jev cannot.** TypeSafe's docs
are explicit that [Jev accepts text only](https://docs.typesafe.ai/concepts/state.md):
images, audio and video are not supported. On every text task here Jev leads;
on this one it does not compete.

Details, including why a single photo costs ~4000 prompt tokens:
**[demos/vision/](demos/vision/)**.

---

## Install

Python 3.10+, `cmake`, a C++ toolchain, ~7 GB of disk.

```bash
git clone https://github.com/NicolaiLassen/open-bonzi-jev.git
cd open-bonzi-jev

./scripts/build_llama_fork.sh                            # ~2 min
python3 scripts/download_model.py Ternary-Bonsai-2-27B   # 5.95 GB

./bonzi/chat.sh --check          # verify runtime and weights match
./openjev/score.sh -q "Is the sky blue?"
```

Want small and fast instead? `Bonsai-8B` is 1.16 GB, needs no fork, and runs
10× the throughput at 64.5%:

```bash
./scripts/get_llama_cpp.sh
python3 scripts/download_model.py Bonsai-8B
./openjev/score.sh -q "Is the sky blue?" --model bonzi/models/Bonsai-8B
```

### Bonsai 2 needs PrismML's llama.cpp fork

Bonsai 2's ternary weights carry a folded-in **Hadamard rotation**, and the
runtime must apply the matching transform to activations. Stock llama.cpp has
no such transform and does not implement the `PTQ1_0` / `PQ2_0` tensor types at
all, so it refuses the file:

```
gguf_init_from_reader: tensor 'token_embd.weight' has invalid ggml type 142. should be in [0, 43)
```

**Good news: every incompatibility we could produce fails loudly, at load, in
well under a second.** We tested the full matrix, and nothing silently produces
garbage. `chat.py --check` confirms a pairing by *actually loading it*, which
is the only authoritative test:

```console
$ ./bonzi/chat.sh --check
model        Ternary-Bonsai-2-27B-PTQ1_0.gguf (5.95 GB)
quantization PTQ1_0  (ggml tensor type 143)
needs fork   yes
llama-server .../llama.cpp-prism/build/bin/llama-server
trial load   PTQ1_0 tensors, gguf v3, loaded successfully
verdict      OK
```

Bonsai 1's `Q1_0` is merged upstream and runs on stock release binaries. See
[bonzi/README.md](bonzi/README.md) for the full compatibility matrix.

---

## Use it

```bash
./openjev/score.sh -q "Is the sky blue?"
./openjev/score.sh -q "Which team?" --options billing,engineering,product \
                   --state '{"subject":"500 errors on every API call"}'
./openjev/score.sh --input openjev/examples/wanli256.jsonl
```

Keep the model resident, because 27B weights are not free to page in:

```bash
./openjev/api.sh          # localhost:8000, no auth
./openjev/demo.sh         # in another terminal
```

`POST /decide`, `POST /chat`, `GET /health`. `demo.py` triages support tickets
and drafts replies only for those judged both angry and urgent. It is a plain
`if angry and urgent:`, where the conditions happen to be semantic.

---

## How the scoring works

1. Build a chat prompt: a system line saying the model is a decision function
   that replies with a letter, then the state, the question, and the options
   lettered `A.`, `B.`, …
2. Apply the chat template with **thinking disabled**, so the assistant turn
   begins exactly where the answer letter goes.
3. One forward pass. Take the next-token scores.
4. Look up each option letter's score (`A`…`Z` are single tokens in these
   vocabularies) and softmax over **just those**.

**Thinking mode is load-bearing.** Bonsai 2 reasons by default at `xhigh`
effort. With `enable_thinking: false` its template emits an already-closed
empty block, `<think>\n\n</think>\n\n`, so the next token is the answer. Leave
it on and the next token is the first word of a monologue, and every
probability is noise. The scorer always disables it.

For GGUF models the prompt is rendered and tokenized **by llama-server**
(`/apply-template`, `/tokenize`, `--jinja`), never by a HuggingFace tokenizer,
so one scorer drives any GGUF whatever its vocabulary. Bonsai 1 8B has 151k
tokens, Bonsai 2 27B has 248k, and `A` is a different id in each.

### Backends

| Backend | For | Notes |
|---|---|---|
| `llama.cpp` | any `.gguf` | default; CPU, Metal, CUDA, Vulkan |
| `mlx` | PrismML's MLX pack | native Apple Silicon, no server process. See [docs/MLX.md](docs/MLX.md) |
| `torch` | HuggingFace checkpoints | `logits_to_keep=1`, BF16, auto MPS/CUDA/CPU |

---

## Layout

```
openjev/    score.py  api.py  demo.py  ggufinfo.py  mlx_backend.py
            jev_backend.py  calibration.py  examples/
bonzi/      chat.py                                 weights + runtimes
demos/      browser/  wikinav.py  wikifind.py  sweep.py  make_video.py
            vision/   vision_score.py  images/
scripts/    build_llama_fork.sh  get_llama_cpp.sh  download_model.py
            benchmark.py  make_charts.py  calibrate.py  stop_servers.sh
docs/       BUILD_NOTES.md  MLX.md  charts/
results/    benchmark.json + per-run JSONL
```

- [docs/BUILD_NOTES.md](docs/BUILD_NOTES.md): measurements, the gotcha table, the corporate-proxy saga
- [bonzi/README.md](bonzi/README.md): the runtime/quantization compatibility matrix
- [openjev/README.md](openjev/README.md): the scorer in detail
- [CREDITS.md](CREDITS.md): who made what

## Reproducing the numbers

```bash
python3 scripts/download_model.py --list
python3 scripts/benchmark.py            # writes results/benchmark.json
python3 scripts/make_charts.py          # regenerates docs/charts/*.svg
```

Method: one fresh llama-server per (model, eval); a warmup pass that is **not**
measured, because the first questions pay for shader compilation, paging in
weights and filling the prompt cache; the same prompt and eval file for every
model; label-balanced eval sets so chance is `1/n_options`.

## Licence

MIT for the code here. Bonsai models are Apache 2.0 from PrismML and are **not
redistributed**. `download_model.py` fetches them from source. SemIf is MIT.
llama.cpp is MIT.
