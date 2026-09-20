#!/usr/bin/env python3
"""
Ask a Bonsai model one question and print the reply, speed and memory use.

Bonsai is PrismML's low-bit model family (https://prismml.com). Two generations
matter here and they need different runtimes:

  Bonsai 2 (ternary, PTQ1_0/PQ2_0)  PrismML's llama.cpp fork -- ternary kernels
                                    plus a Hadamard activation transform that
                                    stock llama.cpp does not have. This is the
                                    default.
  Bonsai 1 (1-bit, Q1_0)            merged upstream, so stock llama.cpp runs it.

Pointing stock llama.cpp at a ternary file is the one failure worth knowing
about: PQ2_0 and PTQ1_0 are rejected as unknown types, but Q2_0 *loads* and
then emits gibberish, because the Hadamard transform never runs. --check
verifies the binary before loading anything.

Usage:
    chat.py "Hello world"
    chat.py --model Bonsai-8B "Hello world"
    chat.py --think --effort medium "Why is the sky blue?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "openjev"))

from ggufinfo import needs_prism_runtime, read_gguf_info  # noqa: E402
from score import (  # noqa: E402  - shares one llama-server client
    LlamaCppBackend,
    find_llama_server,
    resolve_gguf,
    _post,
)

DEFAULT_MODEL = "Ternary-Bonsai-2-27B"


def check_runtime(gguf: Path, exe: str, trial: bool = True) -> tuple[bool, str]:
    """Can this llama-server actually run this file?

    Answered by loading it, which is the only authoritative test. The old
    version guessed -- it scanned the binary and its shared libraries for
    ternary type names -- and guessing was wrong in both directions: PrismML's
    build does not identify itself in --version, and a renamed .gguf defeated
    the filename check that decided whether the scan even mattered.

    Loading is also cheap to fail: every incompatibility we could produce is
    rejected in the GGUF header, in well under a second, before any weights
    are read.
    """
    from score import LlamaCppBackend, diagnose_load_failure

    info = read_gguf_info(gguf)
    prism = needs_prism_runtime(info)
    desc = f"{info['name']} tensors, gguf v{info['version']}"

    if not trial:
        return (not prism), desc + " (not verified -- no trial load)"

    backend = None
    try:
        backend = LlamaCppBackend(gguf, gpu=True, ctx=512, llama_server=exe)
        return True, desc + ", loaded successfully"
    except SystemExit as e:
        return False, desc + "\n" + str(e)
    finally:
        if backend is not None:
            backend.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="ask a Bonsai model one question")
    ap.add_argument("message", nargs="*", help="the prompt")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"name under models/, or a path to a .gguf "
                         f"(default {DEFAULT_MODEL})")
    ap.add_argument("--cpu", action="store_true", help="no GPU offload")
    ap.add_argument("--think", action="store_true",
                    help="enable reasoning (off by default -- Bonsai 2 thinks "
                         "at xhigh effort by default and it is slow)")
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"], default="medium",
                    help="reasoning effort when --think (default medium)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="these models advertise 262K; 4096 keeps the KV cache sane")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--llama-server")
    ap.add_argument("--check", action="store_true",
                    help="verify this runtime can load this model, and exit")
    ap.add_argument("--no-trial", action="store_true",
                    help="with --check, inspect the header but skip the load")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    model_arg = Path(args.model)
    if not model_arg.exists():
        model_arg = HERE / "models" / args.model
    gguf = resolve_gguf(model_arg)

    exe = find_llama_server(args.llama_server)
    info = read_gguf_info(gguf)
    prism = needs_prism_runtime(info)

    if args.check:
        print(f"model        {gguf.name} ({gguf.stat().st_size / 1e9:.2f} GB)")
        print(f"quantization {info['name']}  (ggml tensor type {info['dominant']})")
        print(f"needs fork   {'yes' if prism else 'no -- stock llama.cpp implements this type'}")
        print(f"llama-server {exe}")
        ok, detail = check_runtime(gguf, exe, trial=not args.no_trial)
        print(f"trial load   {detail}")
        print(f"verdict      {'OK' if ok else 'INCOMPATIBLE'}")
        return 0 if ok else 1

    if prism:
        # Not fatal by itself -- the load below is the real test -- but say so
        # up front, because this is the usual reason it is about to fail.
        pass

    message = " ".join(args.message).strip()
    if not message:
        raise SystemExit("nothing to ask")

    t_load = time.perf_counter()
    backend = LlamaCppBackend(
        gguf, gpu=not args.cpu, ctx=args.ctx,
        llama_server=args.llama_server, verbose=args.verbose,
    )
    load_s = time.perf_counter() - t_load

    where = "CPU" if args.cpu else "GPU (Metal/Vulkan/CUDA)"
    print(f"[{gguf.name} on {where}, loaded in {load_s:.1f} s]", file=sys.stderr)

    try:
        kwargs = {"enable_thinking": bool(args.think)}
        if args.think:
            kwargs["reasoning_effort"] = args.effort
        prompt = _post(
            f"{backend.url}/apply-template",
            {
                "messages": [{"role": "user", "content": message}],
                "chat_template_kwargs": kwargs,
            },
        )["prompt"]

        out = _post(
            f"{backend.url}/completion",
            {
                "prompt": prompt,
                "n_predict": args.max_tokens,
                "temperature": args.temp,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
        )
        text = out.get("content", "")
        reasoning, reply = split_thinking(text)
        if reasoning and args.think:
            print(f"\n[thinking]\n{reasoning}\n", file=sys.stderr)
        print(reply.strip())
        print(format_timings(out), file=sys.stderr)
    finally:
        backend.close()
    return 0


def split_thinking(text: str) -> tuple[str, str]:
    """Separate a <think>...</think> block from the reply. With thinking off
    the template pre-closes an empty block, which shows up here as an empty
    reasoning string."""
    m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if not m:
        return "", text
    return m.group(1).strip(), text[m.end():]


def format_timings(out: dict) -> str:
    t = out.get("timings") or {}
    pn, pms = t.get("prompt_n"), t.get("prompt_ms")
    gn, gms = t.get("predicted_n"), t.get("predicted_ms")
    bits = []
    if pn and pms:
        bits.append(f"prompt {pn} tok at {pn / (pms / 1000):.1f} tok/s")
    if gn and gms:
        bits.append(f"generated {gn} tok at {gn / (gms / 1000):.1f} tok/s")
    if pms or gms:
        bits.append(f"{((pms or 0) + (gms or 0)) / 1000:.1f} s total")
    return "[" + ", ".join(bits) + "]" if bits else ""


if __name__ == "__main__":
    sys.exit(main())
