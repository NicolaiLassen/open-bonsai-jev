# MLX on Apple Silicon

*"Why don't you run it with MLX?"* Fair question. We do now, and the answer
turned out to be more interesting than expected.

On a Mac there are two ways to run Bonsai 2's ternary weights:

| | Path | Needs |
|---|---|---|
| **llama.cpp** | GGUF behind `llama-server` | [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), built with Metal |
| **MLX** | the published MLX pack, in-process | `mlx` + `mlx-lm` + `mlx-vlm`, and the pack's own loader |

## Measured, same machine, same 24 questions

Apple M4 Pro, Bonsai 2 27B, WANLI rows, 3-question warmup excluded.

| Backend | Accuracy | Throughput | Median forward |
|---|---|---|---|
| llama.cpp (fork, Metal) | 15/24 | 29 q/min | 2.250 s |
| **MLX (pack runtime)** | 15/24 | **30 q/min** | **1.945 s** |

Agreement between the two: **24/24 identical choices**, mean absolute
difference in reported probability **0.0006** (max 0.0054).

Two conclusions:

1. **They are the same model, and the implementations agree.** Two independent
   runtimes, with different kernels, a different tokenizer path and different
   quantization unpacking, landing on the same answers with probabilities
   matching to three decimal places is the strongest correctness check in this
   repo. It also means `score.py`'s prompt and letter-lookup are not doing
   anything backend-specific.
2. **Neither is meaningfully faster.** MLX edges it on per-forward latency;
   llama.cpp claws it back with prompt-prefix caching across questions that
   share a system prompt. Pick on ergonomics, not speed.

> One caution on measuring this yourself: `score.py` buffers its JSONL output,
> so watching `wc -l` on the results file badly understates progress. Our first
> read of the MLX run suggested ~3 q/min; it was actually ~30. Time the process,
> don't count lines.

## Setup

The MLX pack is **not** stock `mlx-lm`. Its weights are 2-bit affine with a
Hadamard transform that ordinary MLX loaders do not apply, so it ships its own
loader in `runtime/`.

```bash
python3 -m venv .venv-mlx
./.venv-mlx/bin/pip install "mlx==0.32.0" "mlx-lm==0.31.3" "mlx-vlm==0.6.3" \
                            "transformers==5.5.0" "numpy>=2.0" "tokenizers>=0.21"

python3 - <<'PY'
# pack lives in bonzi/models/Ternary-Bonsai-2-27B-mlx (8.0 GB)
PY
./.venv-mlx/bin/python openjev/score.py -q "Is the sky blue?" \
    --model bonzi/models/Ternary-Bonsai-2-27B-mlx
```

`score.py` picks the MLX backend automatically when the model directory
contains `runtime/vision_artifact.py`; `--backend mlx` forces it.

## Two traps

**The bundled text-only loader rejects the pack.** `runtime/artifact.py`
exposes `load_model()` and is what `PACK-RUNTIME.md` documents, but it checks
`schema_version == 1` while the published pack is `schema_version: 2` with a
vision tower:

```
ValueError: Unsupported packed model schema
```

Use `vision_artifact.load_vl_model()` instead and take `.language_model`. That
is what `openjev/mlx_backend.py` does. Scoring is text-only, so the vision
tower is loaded and then ignored.

**It needs `mlx-vlm`,** which the text-only path does not, on top of `mlx` and
`mlx-lm`.

## Why it suits a scorer

MLX hands you the logits directly:

```python
out = model(mx.array([ids]))
logits = out.logits[0, -1]          # the whole next-token distribution
```

No HTTP hop, no server process to orphan, and no `n_probs` truncation. The
llama.cpp path only sees the top 20 tokens, so an option letter outside that
window counts as probability zero. With MLX every option letter is always in
range. It has not mattered yet (`label_mass` stays above 0.98 in practice) but
it is one less approximation.

## Not done here

- **Vision.** The pack includes a tower; we never feed it an image.
- **Prompt-prefix reuse.** llama.cpp gets this via `cache_prompt`; the MLX
  backend recomputes the full prompt every call. Caching the shared system
  prefix is the obvious next speed-up on this path.
- **The MLX builds of Bonsai 1** (`prism-ml/Bonsai-*-mlx-1bit`).
