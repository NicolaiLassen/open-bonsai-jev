---
license: mit
base_model:
  - prism-ml/Ternary-Bonsai-2-27B-gguf
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
  - name: bonzi-27b-v2-jev
    results:
      - task:
          type: natural-language-inference
          name: Typed decision (3-way NLI)
        dataset:
          type: alisawuffles/WANLI
          name: WANLI-256 (seeded label-balanced subset)
        metrics:
          - type: accuracy
            value: 0.7461
            name: Accuracy
      - task:
          type: text-classification
          name: Typed decision (binary smoke test)
        dataset:
          type: custom
          name: easy100
        metrics:
          - type: accuracy
            value: 1.0000
            name: Accuracy
---

# bonzi-27b-v2-jev

**Bonsai 2 27B as a typed decision function.** No weights here. This is a
recipe and a measurement for running
[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) as a
Jev-style System One model: one forward pass in, a calibrated probability per
option out.

Asked `"Is the sky blue?"`, a real row from this model's own run:

```json
{"choice": "true", "probabilities": {"true": 0.99331, "false": 0.00669},
 "label_mass": 0.992448, "prompt_tokens": 95, "forward_s": 1.494}
```

Code: **https://github.com/NicolaiLassen/open-bonzi-jev**

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
| Weights | [`Ternary-Bonsai-2-27B-PTQ1_0.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) |
| Quantization | ternary (`PTQ1_0`) |
| On disk | **5.95 GB** |
| Runtime | PrismML fork |
| WANLI-256 | **74.6%** (chance 33.3%) |
| easy100 | 100% |
| Throughput | 24 decisions/min |
| Median latency | 2443 ms |
| Median `label_mass` | 0.993 |

> **This model needs [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp).**
> Its ternary weights carry a folded-in Hadamard rotation that stock llama.cpp
> cannot apply, and it does not implement the tensor type at all. Stock
> refuses the file at load with `invalid ggml type`. Build the fork with
> `scripts/build_llama_fork.sh`.

### Against the rest of the family

Apple M4 Pro, GPU, ctx 4096, warmup excluded. This model ranks **#1 of
6** on decision quality.

| Model | On disk | WANLI-256 | decisions/min |
|---|---|---|---|
| Bonsai 1 1.7B | 0.25 GB | 52.0% | 925 |
| Bonsai 1 4B | 0.57 GB | 60.2% | 398 |
| Bonsai 1 8B | 1.16 GB | 64.5% | 230 |
| Bonsai 1 27B | 3.80 GB | 71.1% | 31 |
| Ternary Bonsai 1 8B | 2.18 GB | 65.2% | 200 |
| **Bonsai 2 27B** | 5.95 GB | 74.6% | 24 |

## Usage

```bash
git clone https://github.com/NicolaiLassen/open-bonzi-jev
cd open-bonzi-jev
./scripts/build_llama_fork.sh
python3 scripts/download_model.py Ternary-Bonsai-2-27B

./openjev/score.sh -q "Is the user angry?" \
    --state '{"msg":"this is broken AGAIN"}' \
    --model bonzi/models/Ternary-Bonsai-2-27B
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
