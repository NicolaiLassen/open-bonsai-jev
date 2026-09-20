# openjev: the mechanism

Semantic-if scoring: ask a typed question, read the answer straight off the
model's next-token scores.

A re-implementation of the "direct" mode from
**[SemIf](https://github.com/TheoLeeCJ/SemIf)** (formerly OpenJev) by
**TheoLeeCJ**. None of their code is used; the prompt here is our own, so the
numbers are not comparable to theirs. See [../CREDITS.md](../CREDITS.md).

## score.py

```bash
python3 score.py -q "Is the sky blue?"
python3 score.py -q "Is the user angry?" --state '{"msg":"broken AGAIN"}'
python3 score.py -q "Which team?" --options billing,engineering,product
python3 score.py --input examples/easy100.jsonl --gpu
```

```json
{"choice": "true", "probabilities": {"true": 0.951217, "false": 0.048783},
 "label_mass": 0.98913, "prompt_tokens": 91, "forward_s": 1.586,
 "backend": "llama.cpp"}
```

`--state` takes a string or JSON. `--options` defaults to `true,false` and
takes up to 26. `--input` reads JSONL with `question`, and optionally `state`,
`options` and `expected`; with `expected` present it reports accuracy.
`--warmup N` (default 5) runs N questions before the clock starts. The first
few pay for shader compilation, paging weights in and filling the prompt cache,
and folding that into the measurement makes small models look far worse than
they are.

Two eval files ship here:

| File | Rows | Chance | What it is |
|---|---|---|---|
| `wanli256.jsonl` | 256 | 33.3% | Seeded, label-balanced subset of [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI). The real measurement. |
| `easy100.jsonl` | 100 | 50% | Trivial true/false facts. A smoke test that saturates above 4B. |
| `simple10.jsonl` | 10 | n/a | State handling and multi-way options. |

## How it works

1. Build a chat prompt: a system line telling the model it is a decision
   function that replies with a letter, the state, the question, and the
   options lettered `A.`, `B.`, …
2. Apply the chat template with **thinking disabled**, so the assistant turn
   starts exactly where the answer letter belongs. On Bonsai 2 that renders as
   `<|im_start|>assistant\n<think>\n\n</think>\n\n`, an empty, pre-closed
   reasoning block. This step is not optional: leave thinking on and the next
   token is the first word of a monologue, and every number below is noise.
3. One forward pass. Take the next-token scores.
4. Look up the score of each option letter (`A`, `B`, … are single tokens in
   every vocabulary here) and softmax over **just those**.

`label_mass` is how much of the whole-vocabulary probability those letters
held. Near 1.0 means the model really was answering with one of the options.
**It does not mean the answer is right.** Bonsai 1 8B answers "does a spider
have two legs?" with `true` at 0.999 and a `label_mass` of 0.9999.

## Backends

| | |
|---|---|
| `llama.cpp` | any `.gguf`. Default when the model looks like one. |
| `mlx` | a PrismML MLX pack, in-process on Apple Silicon. Auto-selected when the model directory has `runtime/vision_artifact.py`. See [../docs/MLX.md](../docs/MLX.md). |
| `torch` | a HuggingFace checkpoint, `logits_to_keep=1`, BF16, auto-selects MPS / CUDA / CPU. |

The llama.cpp and MLX backends were cross-checked on the same 24 questions:
**24/24 identical choices**, mean absolute probability difference 0.0006. Two
independent runtimes agreeing that closely is the best evidence we have that
the scorer itself is correct.

For GGUF the prompt is rendered and tokenized **by llama-server itself**
(`/apply-template` and `/tokenize`, started with `--jinja`), never by a
HuggingFace tokenizer. That is what lets one scorer drive any GGUF whatever its
vocabulary. Bonsai 1 8B is Qwen3-based with 151k tokens, Bonsai 2 27B has
248k, and the token id of `A` differs. `--tokenizer <dir>` forces the
HuggingFace path if you want byte-identical prompts across backends.

llama.cpp probabilities come from `/completion` with `n_predict: 1,
n_probs: 20`. They are a plain softmax over the raw scores, unaffected by
sampling settings. A letter outside the top 20 counts as zero.

## Calibration

`label_mass` says the model answered with one of your options. It does not say
the probability is worth believing, and by default it is not: Bonsai 2 27B
averages 0.899 confidence at 74.6% accuracy, and in the bin where it claims
0.99 or more it is right 86% of the time.

`--temperature T` divides the option logits before the softmax, and
`--calibrated` looks up the fitted T for the model:

```bash
python3 ../scripts/calibrate.py            # fit from scored eval files
python3 score.py -q "Is the sky blue?" --calibrated
python3 score.py --input examples/wanli256.jsonl --temperature 2.45
```

This never changes `choice`: a positive divisor preserves the ordering of the
options, so accuracy is untouched and only the reported probability moves.
Verified across every model and row in this repo, 0 changes in 1,858. See
[calibration.py](calibration.py) and the table in the
[root README](../README.md#calibration-the-probabilities-lie-and-one-number-fixes-it).

## ggufinfo.py

Reads a GGUF header and reports what the file actually is, so nothing depends
on filenames:

```console
$ python3 ggufinfo.py ../bonzi/models/*/*.gguf
Bonsai-8B-Q1_0.gguf                Q1_0     gguf v3  Q1_0x254, F32x145
Ternary-Bonsai-2-27B-PTQ1_0.gguf   PTQ1_0   gguf v3  PTQ1_0x402, F32x353, BF16x96  [needs PrismML fork]
```

## api.py

One loaded model, several tasks. Loading Bonsai 2 27B costs seconds and ~6 GB;
`score.py` per question pays that every time.

```bash
python3 api.py --model ../bonzi/models/Ternary-Bonsai-2-27B --gpu
python3 demo.py            # in another terminal
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | n/a | model, backend, endpoints |
| `POST /decide` | `{"state":…, "question":…, "options":[…]}` | the scorer's JSON |
| `POST /chat` | `{"message":…}` or `{"messages":[…]}`, `max_tokens`, `think`, `effort` | `{"reply","reasoning","timings"}` |

It binds to localhost and has **no authentication**. `--host 0.0.0.0` would put
an unauthenticated model server on your network.

`demo.py` triages three support tickets, three decisions each, and drafts a
reply only where the ticket is judged both angry and urgent. It is ordinary
`if angry and urgent:` control flow, where the conditions happen to be
semantic.
