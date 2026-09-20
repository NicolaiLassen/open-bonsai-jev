#!/usr/bin/env python3
"""
Generate one HuggingFace model card per Bonsai configuration we measured.

Each card is a *recipe*, not weights: it documents running that specific Bonsai
model as a Jev-style typed decision function, and carries that model's own
measured numbers. No weights are redistributed.

    make_model_cards.py            # -> huggingface/<repo-name>/README.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "huggingface"
GH = "https://github.com/NicolaiLassen/open-bonzi-jev"

# folder -> (hf repo slug, upstream weights repo, upstream file, quant blurb)
CARDS = {
    "Bonsai-1.7B": ("bonzi-1.7b-v1-jev", "prism-ml/Bonsai-1.7B-gguf",
                    "Bonsai-1.7B-Q1_0.gguf", "1-bit (`Q1_0`)"),
    "Bonsai-4B": ("bonzi-4b-v1-jev", "prism-ml/Bonsai-4B-gguf",
                  "Bonsai-4B-Q1_0.gguf", "1-bit (`Q1_0`)"),
    "Bonsai-8B": ("bonzi-8b-v1-jev", "prism-ml/Bonsai-8B-gguf",
                  "Bonsai-8B-Q1_0.gguf", "1-bit (`Q1_0`)"),
    "Bonsai-27B": ("bonzi-27b-v1-jev", "prism-ml/Bonsai-27B-gguf",
                   "Bonsai-27B-Q1_0.gguf", "1-bit (`Q1_0`)"),
    "Ternary-Bonsai-8B": ("bonzi-8b-ternary-v1-jev", "prism-ml/Ternary-Bonsai-8B-gguf",
                          "Ternary-Bonsai-8B-PQ2_0.gguf", "ternary (`PQ2_0`)"),
    "Ternary-Bonsai-2-27B": ("bonzi-27b-v2-jev", "prism-ml/Ternary-Bonsai-2-27B-gguf",
                             "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "ternary (`PTQ1_0`)"),
}


def real_example(m: dict) -> str:
    """A genuine scored row from this model's own run -- not a mock-up."""
    import json as _json
    f = ROOT / "results" / f"{Path(m['file']).stem}.easy100.jsonl"
    try:
        rows = [_json.loads(l) for l in f.open(encoding="utf-8")]
    except OSError:
        return ""
    r = rows[0]
    probs = ", ".join(f'"{k}": {v}' for k, v in r["probabilities"].items())
    return (f'{{"choice": "{r["choice"]}", "probabilities": {{{probs}}},\n'
            f' "label_mass": {r["label_mass"]}, "prompt_tokens": {r["prompt_tokens"]}, '
            f'"forward_s": {r["forward_s"]}}}')


def card(m: dict, slug: str, upstream: str, fname: str, quant: str, allm: list) -> str:
    e = m["evals"]
    wanli, easy = e["wanli256"], e["easy100"]
    fork = m["runtime"] != "stock llama.cpp"
    rank = sorted(allm, key=lambda x: -x["evals"]["wanli256"]["accuracy"])
    place = [x["label"] for x in rank].index(m["label"]) + 1

    rows = "\n".join(
        f"| {'**' if x['label']==m['label'] else ''}{x['label']}"
        f"{'**' if x['label']==m['label'] else ''} | {x['size_gb']:.2f} GB | "
        f"{x['evals']['wanli256']['accuracy']*100:.1f}% | "
        f"{x['evals']['wanli256']['questions_per_min']:.0f} |"
        for x in allm)

    runtime_note = (
        f"""> **This model needs [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp).**
> Its ternary weights carry a folded-in Hadamard rotation that stock llama.cpp
> cannot apply, and it does not implement the tensor type at all. Stock
> refuses the file at load with `invalid ggml type`. Build the fork with
> `scripts/build_llama_fork.sh`."""
        if fork else
        """> Runs on **stock llama.cpp**. Bonsai 1's `Q1_0` is merged upstream, so
> ordinary release binaries work, and no fork is required."""
    )

    return f"""---
license: mit
base_model:
  - {upstream}
tags:
  - semantic-if
  - typed-decisions
  - system-one
  - logprobs
  - llama-cpp
  - gguf
  - bonsai
  - jev
language:
  - en
pipeline_tag: text-classification
library_name: llama.cpp
model-index:
  - name: {slug}
    results:
      - task:
          type: natural-language-inference
          name: Typed decision (3-way NLI)
        dataset:
          type: alisawuffles/WANLI
          name: WANLI-256 (seeded label-balanced subset)
        metrics:
          - type: accuracy
            value: {wanli['accuracy']:.4f}
            name: Accuracy
      - task:
          type: text-classification
          name: Typed decision (binary smoke test)
        dataset:
          type: custom
          name: easy100
        metrics:
          - type: accuracy
            value: {easy['accuracy']:.4f}
            name: Accuracy
---

# {slug}

**{m['label']} as a typed decision function.** No weights here. This is a
recipe and a measurement for running
[`{upstream}`]({upstream and 'https://huggingface.co/' + upstream}) as a
Jev-style System One model: one forward pass in, a calibrated probability per
option out.

Asked `"Is the sky blue?"`, a real row from this model's own run:

```json
{real_example(m)}
```

Code: **{GH}**

## Where this comes from

1. **[TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**
   shipped **Jev** and named the category: models returning typed
   probabilistic decisions instead of text.
2. **[TheoLeeCJ](https://github.com/TheoLeeCJ)** open-sourced the mechanism as
   **[SemIf](https://github.com/TheoLeeCJ/SemIf)**, *originally released as
   OpenJev*, reading option logits straight out of a frozen open model.
3. **This** ports that onto **[PrismML's Bonsai](https://prismml.com)** low-bit
   models.

`score.py` is an **independent re-implementation** of SemIf's "direct" mode
with its own prompt. No SemIf code is used, so **these numbers are not
comparable to SemIf's published benchmarks.** Not affiliated with TypeSafe,
PrismML, or TheoLeeCJ.

## This model

| | |
|---|---|
| Weights | [`{fname}`](https://huggingface.co/{upstream}/blob/main/{fname}) |
| Quantization | {quant} |
| On disk | **{m['size_gb']:.2f} GB** |
| Runtime | {m['runtime']} |
| WANLI-256 | **{wanli['accuracy']*100:.1f}%** (chance 33.3%) |
| easy100 | {easy['accuracy']*100:.0f}% |
| Throughput | {wanli['questions_per_min']:.0f} decisions/min |
| Median latency | {wanli['median_forward_s']*1000:.0f} ms |
| Median `label_mass` | {wanli['median_label_mass']:.3f} |

{runtime_note}

### Against the rest of the family

Apple M4 Pro, GPU, ctx 4096, warmup excluded. This model ranks **#{place} of
{len(allm)}** on decision quality.

| Model | On disk | WANLI-256 | decisions/min |
|---|---|---|---|
{rows}

## Usage

```bash
git clone {GH}
cd open-bonzi-jev
{'./scripts/build_llama_fork.sh' if fork else './scripts/get_llama_cpp.sh'}
python3 scripts/download_model.py {[k for k, v in CARDS.items() if v[0] == slug][0]}

./openjev/score.sh -q "Is the user angry?" \\
    --state '{{"msg":"this is broken AGAIN"}}' \\
    --model bonzi/models/{[k for k, v in CARDS.items() if v[0] == slug][0]}
```

Options default to `true,false`; pass `--options a,b,c` for up to 26.

## How it works

A lettered multiple-choice prompt is built from the state, question and
options. The chat template is applied with **thinking disabled**, so the
assistant turn begins exactly where the answer letter goes. One forward pass
runs, and the scorer softmaxes over just the option-letter tokens.

## Limitations

- **`label_mass` is not confidence in correctness.** It reports how much
  probability mass landed on the option letters, meaning you got *an* answer,
  not a right one. Bonsai 1 8B answers *"does a spider have two legs?"* with `true`
  at **0.998** confidence and `label_mass` 0.9997.
- **Probabilities are conditional on the options you supplied,** and are not
  calibrated for your workload. Validate on your own data before branching on a
  threshold.
- **WANLI-256 is 256 rows.** Treat the difference between neighbouring models
  as indicative, not decisive.
- **Text only.** The 27B packs include a vision tower that is never used here.

## Licence

MIT for the code. Bonsai weights are Apache 2.0 from PrismML and are **not
redistributed**. The downloader fetches them from source.
"""


def main():
    src = ROOT / "results" / "benchmark.json"
    if not src.exists():
        raise SystemExit("run scripts/benchmark.py first")
    data = json.loads(src.read_text())
    allm = [m for m in data["models"] if m.get("evals", {}).get("wanli256", {}).get("accuracy")]

    for m in allm:
        spec = CARDS.get(m["folder"])
        if not spec:
            continue
        slug, upstream, fname, quant = spec
        d = OUT / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(card(m, slug, upstream, fname, quant, allm), encoding="utf-8")
        print(f"wrote huggingface/{slug}/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
