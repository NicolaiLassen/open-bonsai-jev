# Credits

open-bonzi-jev builds on work by TypeSafe, TheoLeeCJ and PrismML. This is the
accounting of what came from where.

## The lineage, in three steps

**1. TypeSafe had the idea and named the category.**
[Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
Models that return typed, probabilistic decisions instead of text, in tens
to hundreds of milliseconds. **Jev** is their closed, hosted implementation.
Every repository in this space, including this one, is downstream of that
framing.

**2. [TheoLeeCJ](https://github.com/TheoLeeCJ) open-sourced the mechanism.**
[SemIf](https://github.com/TheoLeeCJ/SemIf), **originally released under the
name OpenJev**, showed that you get the same interface from *frozen open
models* by reading option logits directly out of one forward pass, no training
and no fine-tuning required. MIT licensed, running in the browser at
[openjev.com](https://openjev.com).

**That project is why this one exists.** It is the reference implementation and
it is better than this one at nearly everything: a browser build, shared-prefix
reuse (`--mode shared`), a reranker comparison, perturbation tests, committed
row-level outputs, pinned model revisions and prompt hashes. If you want to
understand or use semantic-if scoring properly, start there.

**3. PrismML made it cheap to run.** [Bonsai](https://prismml.com) is their
low-bit family: a 27B in 5.95 GB, a usable model in 0.25 GB, Apache 2.0. Used
exactly as published here, with nothing retrained or requantized.

**What this repo adds** is the combination of the second and the third, plus
the measurements: hosted Jev benchmarked on the same eval files, a 110-pair
browser-agent sweep, a calibration fit, the runtime compatibility matrix, and
the vision demo.

### What is and is not ours

`openjev/score.py` is an **independent re-implementation** of SemIf's "direct"
mode, written from its public description, with **our own prompt**. No SemIf
code is vendored or copied.

Because the prompt differs, **numbers measured here are not directly comparable
to SemIf's published benchmarks.** Where this repo cites SemIf's 0.637 on
WANLI-256 next to our own WANLI-256 numbers, that is an order-of-magnitude
sanity check on different rows with a different prompt and a different model.
It is not a like-for-like comparison, and certainly not a claim to beat them. Any
weakness in our numbers is ours.

SemIf states it is independent of and not affiliated with TypeSafe or Jev.
Neither is this.

## The weights: Bonsai, by PrismML

Used exactly as published. We do not train, fine-tune, quantize or
redistribute any weights; `scripts/download_model.py` fetches them from source.

- [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf): Apache 2.0, built on Qwen3.8-27B
- [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) and the rest of the Bonsai 1 family: Apache 2.0, built on Qwen3
- [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit): the MLX pack, including the `runtime/` loader we call
- [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp): the fork carrying the ternary kernels and the Hadamard activation runtime, without which Bonsai 2 does not run at all

## Runtime and tooling

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): Georgi Gerganov and contributors, MIT. `score.py` speaks to `llama-server` over HTTP and leans on `/apply-template`, `/tokenize`, and `n_probs` on `/completion`.
- [Qwen](https://github.com/QwenLM): Alibaba. The base of every Bonsai model here.
- [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI): Liu et al., worker-and-AI NLI. Our `wanli256.jsonl` is a seeded, label-balanced 256-row subset of its public test split.
- [MLX](https://github.com/ml-explore/mlx): Apple. The native Apple Silicon path.
- [transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org): the optional `torch` backend.

## The name

"open-bonzi-jev" contracts **openjev** (the mechanism, from TheoLeeCJ) and
**Bonsai** (the weights, from PrismML), misspelled with a z because that is
what it got called on the machine it was first built on. It is not an official
anything.

Jev and TypeSafe are referenced only to describe what kind of thing this is.
This project is not affiliated with, endorsed by, or connected to TypeSafe,
PrismML, or TheoLeeCJ.
