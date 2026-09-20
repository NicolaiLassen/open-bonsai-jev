#!/usr/bin/env python3
"""
MLX backend: score Bonsai 2 natively on Apple Silicon.

Why this exists at all. On a Mac there are two ways to run Bonsai 2's ternary
weights:

  llama.cpp + PrismML's fork   a GGUF behind llama-server (see score.py)
  MLX + PrismML's pack runtime this file

MLX is the native Apple Silicon path and gives the logits directly, with no
HTTP hop and no server process, which suits a scorer that wants exactly one
number per forward pass.

The catch is that this is not stock `mlx-lm`. The published MLX pack carries
its own loader in `runtime/`, because the weights are 2-bit affine with a
Hadamard transform that ordinary MLX loaders do not apply. Two wrinkles we hit:

  * the pack is `schema_version: 2` and includes a vision tower, so the
    bundled *text-only* `artifact.load_model` rejects it -- use
    `vision_artifact.load_vl_model` and take `.language_model`;
  * it needs `mlx-vlm` in addition to `mlx` and `mlx-lm`.

Install into a separate venv from `runtime/requirements.txt` in the pack.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


class MlxBackend:
    """A PrismML MLX pack, scored by reading the next-token logits."""

    name = "mlx"

    def __init__(self, pack_dir, verbose: bool = False):
        pack = Path(pack_dir)
        runtime = pack / "runtime"
        if not (runtime / "vision_artifact.py").exists():
            raise SystemExit(
                f"{pack} does not look like a PrismML MLX pack "
                f"(no runtime/vision_artifact.py)"
            )
        sys.path.insert(0, str(runtime))

        try:
            import mlx.core as mx
        except ImportError as e:
            raise SystemExit(
                "mlx is not installed. Create a venv and install the pack's "
                "runtime/requirements.txt (mlx, mlx-lm, mlx-vlm, transformers)."
            ) from e

        from vision_artifact import load_vl_model

        self.mx = mx
        t0 = time.perf_counter()
        model, processor, config = load_vl_model(pack)
        self.load_s = time.perf_counter() - t0
        # The pack ships a vision tower we never use; scoring is text only.
        self.model = getattr(model, "language_model", model)
        self.config = config
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self._letter_ids: dict[str, int] = {}
        if verbose:
            print(f"[mlx pack loaded in {self.load_s:.1f}s]", file=sys.stderr)

    def letter_id(self, letter: str) -> int:
        if letter not in self._letter_ids:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"option letter {letter!r} is {len(ids)} tokens here, not 1"
                )
            self._letter_ids[letter] = ids[0]
        return self._letter_ids[letter]

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def score(self, messages: list[dict], options: list[str]) -> dict:
        from score import LETTERS

        mx = self.mx
        prompt = self.render(messages)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        t0 = time.perf_counter()
        out = self.model(mx.array([ids]))
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[0, -1].astype(mx.float32)
        probs = mx.softmax(last)
        mx.eval(last, probs)
        forward_s = time.perf_counter() - t0

        raw, mass = {}, 0.0
        for i, _ in enumerate(options):
            letter = LETTERS[i]
            tid = self.letter_id(letter)
            raw[letter] = float(last[tid])
            mass += float(probs[tid])
        return {
            "logits": raw,
            "label_mass": mass,
            "prompt_tokens": len(ids),
            "forward_s": forward_s,
        }

    def close(self):
        pass
