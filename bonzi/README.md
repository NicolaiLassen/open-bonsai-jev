# bonzi: weights and runtimes

Holds the [Bonsai](https://prismml.com) models and the llama.cpp builds that
execute them. `openjev/score.py` points at whatever lands in `models/`.

## Compatibility matrix

Every cell below was tested on this machine, not inferred from documentation.
Reproduce with `python3 ../openjev/ggufinfo.py models/*/*.gguf` and
`python3 chat.py --check`.

| File | ggml type | Stock llama.cpp | PrismML fork |
|---|---|---|---|
| `Q1_0`, Bonsai 1 | 41 | ✅ runs | ✅ runs |
| `Q2_0` group-64, Ternary Bonsai 1 | 42 | ✅ runs | ✅ runs |
| `Q2_0` legacy group-128 | 42 | ❌ fails at load | ❌ fails at load |
| `PQ2_0`, Bonsai 2 | 142 | ❌ fails at load | ✅ runs |
| `PTQ1_0`, Bonsai 2 | 143 | ❌ fails at load | ✅ runs |

Three findings from building that table, each of which contradicts something
we believed going in:

**1. Nothing silently produces garbage.** Every incompatible pairing fails
during header parsing, in under a tenth of a second, with a message. We went
looking for a silent-gibberish mode, widely described for `Q2_0`, and could
not produce one on current builds. That is why there is no longer any
pre-flight heuristic in this repo: the load *is* the test.

**2. The fork is not needed for Ternary Bonsai 1.** On the group-64 `Q2_0`
file, stock llama.cpp and the fork returned **bit-identical results**: 100/100
identical choices across the eval set, mean probability difference 0.0000. The
fork earns its place only for Bonsai 2's `PQ2_0` and `PTQ1_0`, which stock does
not implement at all.

**3. The two `Q2_0` layouts are indistinguishable by type id.** Both report
ggml type 42. The legacy group-128 layout fails with an offset mismatch; the
fork at least tells you why:

```
this file matches the legacy Prism Q2_0 layout (group size 128 stored as ggml
type id 42), but this build reads Q2_0 as the official group-64 format
```

## Why Bonsai 2 needs the fork

Bonsai 2 folds a blockwise **Hadamard rotation** into its stored weights, and
the runtime must apply the matching transform to activations. Stock llama.cpp
has no such transform and rejects the tensor types outright:

```
gguf_init_from_reader: tensor 'token_embd.weight' has invalid ggml type 142. should be in [0, 43)
```

## Setup

```bash
../scripts/build_llama_fork.sh              # ~2 min; Metal/CUDA/Vulkan auto-detected
python3 ../scripts/download_model.py Ternary-Bonsai-2-27B
python3 chat.py "Hello world"
```

Bonsai 1, which needs no fork:

```bash
../scripts/get_llama_cpp.sh
python3 ../scripts/download_model.py Bonsai-8B
python3 chat.py --model Bonsai-8B "Hello world"
```

## chat.py

```bash
python3 chat.py "Why is the sky blue?"        # thinking off, fast
python3 chat.py --think --effort medium "..." # thinking on
python3 chat.py --cpu "..."                   # no GPU offload
python3 chat.py --check                       # verify runtime ↔ weights
python3 chat.py --check --no-trial            # header only, no load
```

`--check` **loads the model**, because that is the only authoritative answer.
An earlier version guessed by scanning the binary and its shared libraries for
ternary type names; that was wrong in both directions. PrismML's build reports
an ordinary llama.cpp version string (`0.2.0-dev`, no mention of prism), and
the filename test that gated the scan was defeated by renaming a `.gguf`.
Header inspection plus a real load replaced both.

Thinking is **off by default**. Bonsai 2 reasons at `xhigh` effort unless told
otherwise, which on a laptop means a long monologue before any answer.

Context is pinned to 4096. These models advertise 262K and will try to allocate
a KV cache for it.

## Housekeeping

`score.py` and `api.py` stop their server on a clean exit and on SIGINT/SIGTERM,
but a SIGKILL orphans it, and an orphaned 27B holds ~7 GB and quietly competes
with whatever you measure next. We lost a benchmark run to exactly that.

```bash
../scripts/stop_servers.sh
```

## Not set up here

- **The vision tower.** Bonsai 2 27B is a VLM; we run text only and never fetch
  `mmproj` (0.63–0.93 GB).
- **The F16 files** (53.8 GB). The point of Bonsai is not needing them.
- **MLX** lives in [../docs/MLX.md](../docs/MLX.md). It works, and it is neck
  and neck with llama.cpp here: 30 vs 29 decisions/min, and 24/24 identical
  choices between the two.
