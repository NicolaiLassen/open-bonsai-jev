# Build notes

How this was put together, what was measured, and the traps. The original
notes were written on a Windows laptop with an integrated AMD GPU behind a
corporate proxy; the repository has since been rebuilt around Bonsai 2 on
Apple Silicon. Both are recorded here because the contrast is most of the
useful information.

---

## 1. Machines

| | Build machine A | Build machine B |
|---|---|---|
| CPU | AMD Ryzen AI 7 PRO 350 (8c/16t, Zen 5) | Apple M4 Pro |
| GPU | Radeon 860M iGPU (Vulkan) | M4 Pro (Metal) |
| RAM | 28 GB | 48 GB |
| OS | Windows 11 x64 | macOS 15 |
| Network | corporate proxy, see §6 | plain internet |
| Models | Bonsai 1, Qwen3.5-4B | Bonsai 2 27B |

Everything runs **locally on CPU or an ordinary GPU**. No datacentre card, no
cloud API.

---

## 2. Measurements

### Machine B: Bonsai 2 27B (ternary PTQ1_0, 5.95 GB, Metal)

100 easy true/false questions through `score.py`. See the root README for the
current table; chat generation runs at about 19–20 tok/s and the model loads
in about 4 s.

### Machine A: Bonsai 1 and Qwen, 100 easy true/false questions

| Model | Backend | On disk | Questions/min | Accuracy |
|---|---|---|---|---|
| Qwen3.5-4B BF16 | torch, CPU | 9.3 GB | 34 | 100/100 |
| Qwen3.5-4B Q4_K_M | llama.cpp, CPU | 3.0 GB | 32 | 100/100 |
| Qwen3.5-4B Q4_K_M | llama.cpp, iGPU | 3.0 GB | 44 | 100/100 |
| Bonsai-8B Q1_0 | llama.cpp, iGPU | 1.16 GB | **52** | 97/100 |

Chat generation on the iGPU: Bonsai 1 8B about 10.6 tok/s, Bonsai 1 27B about
2.3 tok/s.

The headline trade-off on machine A: the 1-bit model was the fastest and
smallest **and confidently wrong** on some trivial questions. Asked "does a
spider have two legs?" it answered `true` at 0.999. Bonsai 2 answers the same question
correctly at 0.996, which is the clearest single reason this repository moved
to it.

---

## 3. Performance findings

Worth knowing before optimising on similar hardware.

- **Quantization did not make the Qwen model faster on the CPU** (1.87 s vs
  1.65 s per question). `llama-bench` was flat at 45–50 prompt tok/s across 4,
  8, 12 and 16 threads and across prompt lengths. Its only real benefit there
  was memory.
- **Batching does not help.** Two questions at once take twice as long each;
  total throughput is unchanged on both CPU and GPU. That rules out a serial
  bottleneck inside the model and points at a fixed chip-level ceiling, most
  likely the laptop's shared power budget.
- **The iGPU was the most efficient use of that budget**: 44 vs 32–34 q/min.
- **A GPU worker and a CPU worker together** reached only 47/min, about 7% over
  GPU alone, for the same reason.
- **The first Vulkan run after install is much slower** (37 s load, 2.9 s per
  question) while shaders compile. Re-measure before concluding anything.
- **Background load matters.** VS Code on two cores moved timings by ~10%.

---

## 4. Bonsai 2 and the fork

Bonsai 2 ships two ternary GGUFs: `PTQ1_0` (5.95 GB, dense trit packing, ~1.75
bits/weight) and `PQ2_0` (7.21 GB, 2-bit slots, faster to unpack on some
hardware). Each weight matrix goes through a blockwise **Hadamard rotation**
before ternary assignment; the rotation is folded into the stored weights
offline, so it costs no extra bits, but the runtime must apply the matching
transform to activations.

Stock llama.cpp has no such transform and does not implement the `PQ2_0`
(ggml tensor type 142) or `PTQ1_0` (type 143) tensor types at all. Hence
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), branch
`prism`, pinned here to tag `prism-b10709-9a9394a`.

### Correction: there is no silent-gibberish failure

Earlier versions of this document, and the first version of this repo's
README, repeated the widely stated claim that stock llama.cpp **loads `Q2_0`
and emits gibberish**, and the whole `--check` design was justified by
preventing that. We finally tested the full matrix on two builds, and **it is
not true on current builds.** Every incompatible pairing fails at load, during
header parsing, in under a tenth of a second, with a message:

| File | ggml type | Stock b11050 | Fork b10709 |
|---|---|---|---|
| `Q1_0` | 41 | runs | runs |
| `Q2_0` group-64 | 42 | runs | runs |
| `Q2_0` legacy group-128 | 42 | fails at load | fails at load |
| `PQ2_0` | 142 | fails at load | runs |
| `PTQ1_0` | 143 | fails at load | runs |

Two further surprises in that table:

- **The fork is not needed for Ternary Bonsai 1.** On the group-64 `Q2_0`
  file, stock and fork produced **bit-identical output**: 100/100 identical
  choices over the eval set, mean probability difference 0.0000. The fork earns
  its keep only for Bonsai 2's types.
- **The legacy `Q2_0` fails on *both*,** because upstream standardised `Q2_0`
  as group-64 while the old Prism file is group-128 under the same type id.
  The fork's error message is the better one:

  ```
  this file matches the legacy Prism Q2_0 layout (group size 128 stored as
  ggml type id 42), but this build reads Q2_0 as the official group-64 format
  ```

Because every failure is loud and instant, the pre-flight heuristic was
removed. `chat.py --check` now reads the GGUF header for the real tensor type
and then **actually loads the model**, which is the only authoritative test.

### What the old check got wrong

It scanned the `llama-server` binary and its shared libraries for the byte
strings `PTQ1_0` / `hadamard`. That was wrong in both directions:

- **False negatives by design were likely.** The fork does not identify itself:
  `llama-server --version` prints `version: 0.2.0-dev (build 1, commit
  9a9394a)`, with no mention of prism. On macOS the type names live in
  `libllama.dylib` and `libggml-metal.dylib`, not the executable. A statically
  linked or stripped build would have defeated the scan entirely.
- **The file side was worse.** Whether the scan even ran was gated on matching
  the *filename* for `PTQ1_0`/`PQ2_0`/`Q2_0`. Renaming a `.gguf` was enough to
  skip the check.
- **And the type table was wrong.** We had recorded `PTQ1_0` as ggml type 141;
  it is **143**. 141 is the `general.file_type` of a `PQ2_0` file, a different
  enum that does not line up with tensor type ids. The consequence was that
  `needs_prism_runtime()` returned `False` for Bonsai 2 27B, the flagship model.
  Reading the header of the real files is what caught it.

By contrast Bonsai 1's `Q1_0` **is** merged upstream, so stock release binaries
run it on CPU, Vulkan, CUDA and Metal.

---

## 5. Thinking mode is not optional to get right

Bonsai 2's chat template thinks by default, at `xhigh` effort. The tail of the
template is where scoring lives or dies:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

With `enable_thinking: false` the reasoning block is emitted already closed and
empty, so the very next token is the answer letter, exactly what the scorer
reads. Leave it at the default and the next token is the first word of a
monologue, and every probability the scorer reports is meaningless. `score.py`
always passes `enable_thinking: false`; `chat.py` does too unless `--think`.

The same flag also suppresses the reasoning-effort system message the template
would otherwise prepend, so the prompt stays as written.

---

## 6. Environment-specific: a corporate proxy (machine A)

Recorded because it cost a day. **With plain internet, skip all of this**;
`scripts/download_model.py` and `scripts/get_llama_cpp.sh` handle the normal
case.

- Proxy `http://127.0.0.1:3128` for external hosts; internal hosts direct-only
  and in `NO_PROXY`.
- `files.pythonhosted.org` blocked → packages from an internal Artifactory PyPI
  mirror, curated per file, so **pin versions the mirror actually serves**.
  Probe wheel URLs with HEAD before installing; newer versions 404.
- HuggingFace *downloads* blocked → weights from **ModelScope**, which mirrors
  the same repositories. The HuggingFace *API* still worked for listing files
  and comparing hashes, so ModelScope's SHA-256s were checked against it.
  `download_model.py --source modelscope` still does this.
- ModelScope throttles **per connection** to 1–2 MB/s → the downloader fetches
  128 MiB ranges over 8 connections, ~9 MB/s. A single stream took over an hour
  for 9.3 GB.
- GitHub release assets blocked → Artifactory's `remote-generic-github` remote,
  and **use the old owner name `ggerganov/llama.cpp`**. Artifactory had cached
  `ggml-org` as a *file*, so every path under it failed with
  `400 Parent ggml-org must be a folder`. GitHub still redirects the old name.
- Python TLS needed `truststore`; `uv` needed `--native-tls`.
- `--link-mode=copy` for uv, because OneDrive-synced folders break hardlinks.

---

## 7. Gotchas

| Symptom | Cause and fix |
|---|---|
| `invalid ggml type 142/143` | Stock llama.cpp on a Bonsai 2 file. Build the fork. `chat.py --check` confirms by loading. |
| `has offset N, expected M` on a `Q2_0` | The legacy group-128 layout. Use the `_Q2_0_g64` or `PQ2_0` build of that model. |
| An orphaned `llama-server` eating GB | A SIGKILLed scorer cannot clean up after itself. `scripts/stop_servers.sh`. |
| Every probability looks like noise | Thinking mode left on. The scored token is the start of a monologue. |
| A model answers with a long reasoning monologue | Same cause, in `chat.py`. Drop `--think`. |
| Machine freezes loading a 27B | Context size. These advertise 262K; the scripts pin 4096. |
| `chat.cmd is not recognized` | PowerShell will not run programs from the current directory. Use `.\chat.cmd`. |
| `UnicodeEncodeError: 'charmap' codec` | Emoji in a reply, cp1252 console. The scripts call `sys.stdout.reconfigure(encoding="utf-8")`. |
| `llama-server` still running after a crash | Only Ctrl+C shuts it down cleanly. `pkill llama-server`, or on Windows `Get-Process llama-server \| Stop-Process`. |
| Downloads crawl at 1–2 MB/s | ModelScope throttles per connection. That is what the segmented downloader is for. |

---

## 8. Deliberately not here

- **SemIf's own code.** Not cloned, by choice. `score.py` is an independent
  re-implementation with a different prompt, so benchmark numbers are not
  comparable. Their shared-prefix reuse (`--mode shared`), reranker mode and
  evaluation harness are not reproduced.
- **Vision.** Bonsai 2 27B is a VLM; we run text only and never fetch `mmproj`.
- **MLX.** Bonsai 2's MLX build needs a custom loader from the model repo's
  `runtime/` directory rather than stock `mlx-lm`, so it would need its own
  backend. The GGUF path already runs on Metal.
- **Speculative decoding, 4-bit KV cache, Open WebUI.**
- **Prompt-prefix reuse in the scorer.** `cache_prompt` is on but unmeasured.
  Moving the question to the end of the prompt would make ~90% of it reusable;
  SemIf reports 4.6x from this. It is the most promising remaining speed-up.
