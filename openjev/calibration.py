#!/usr/bin/env python3
"""
Calibration: make the reported probability mean what it says.

These models answer well and then overstate how sure they are. On WANLI-256,
Bonsai 2 27B is right 74.6% of the time while averaging 0.899 confidence, and
in the bin where it claims 0.99 or more it is right 86% of the time. Bonsai 1
8B is worse: it claims 0.997 and is right 78%. That is the same defect behind
"does a spider have two legs? true, 0.998".

The fix is one number. Divide the option logits by a temperature T before the
softmax we already run:

    p = softmax(logits / T)

T > 1 flattens the distribution. **It cannot change the answer**, because
dividing by a positive constant preserves the ordering, so accuracy is
untouched. What changes is whether the probability is worth believing, which
is what makes confidence gating possible at all.

An equally simple alternative is additive smoothing, p' = (1-e)p + e/n, and on
some models it wins. Both are supported; `scripts/calibrate.py` fits either
from a scored eval file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Bins for the reliability diagram and the expected calibration error. Narrow
# at the top because that is where these models live and where being wrong
# costs the most.
BINS = [(0.0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .99), (.99, 1.01)]

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "results" / "calibration.json"


def temper(logits: dict[str, float], T: float) -> dict[str, float]:
    """Divide logits by T. Identity when T == 1."""
    if T == 1.0:
        return logits
    return {k: v / T for k, v in logits.items()}


def smooth(probs: dict[str, float], eps: float) -> dict[str, float]:
    """Mix in a little uniform mass."""
    if not eps:
        return probs
    n = len(probs) or 1
    return {k: (1 - eps) * v + eps / n for k, v in probs.items()}


def ece(pairs: list[tuple[float, bool]]) -> float:
    """Expected calibration error: mean gap between claimed and observed,
    weighted by how many predictions land in each bin."""
    if not pairs:
        return 0.0
    total = 0.0
    for lo, hi in BINS:
        chunk = [(c, ok) for c, ok in pairs if lo <= c < hi]
        if not chunk:
            continue
        acc = sum(ok for _, ok in chunk) / len(chunk)
        conf = sum(c for c, _ in chunk) / len(chunk)
        total += len(chunk) / len(pairs) * abs(acc - conf)
    return total


def reliability(pairs: list[tuple[float, bool]]) -> list[dict]:
    """Per-bin counts, mean claimed probability, and observed accuracy."""
    out = []
    for lo, hi in BINS:
        chunk = [(c, ok) for c, ok in pairs if lo <= c < hi]
        out.append({
            "lo": lo, "hi": min(hi, 1.0), "n": len(chunk),
            "confidence": (sum(c for c, _ in chunk) / len(chunk)) if chunk else None,
            "accuracy": (sum(ok for _, ok in chunk) / len(chunk)) if chunk else None,
        })
    return out


def pairs_from_rows(rows: list[dict], T: float = 1.0, eps: float = 0.0):
    """(confidence, correct) for each scored row, after optional calibration.

    Rows come from score.py --output, which stores probabilities rather than
    logits. Probabilities over the options are a softmax, so log p recovers the
    logits up to a constant, which is all a temperature needs.
    """
    out = []
    for r in rows:
        probs = r["probabilities"]
        if T != 1.0:
            lg = {k: (math.log(v) if v > 0 else -50.0) / T for k, v in probs.items()}
            m = max(lg.values())
            ex = {k: math.exp(v - m) for k, v in lg.items()}
            z = sum(ex.values()) or 1.0
            probs = {k: v / z for k, v in ex.items()}
        if eps:
            probs = smooth(probs, eps)
        top = max(probs, key=probs.get)
        out.append((probs[top], top == r.get("expected")))
    return out


def fit_temperature(rows: list[dict], lo=0.5, hi=6.0, step=0.05) -> tuple[float, float]:
    """The T that minimises ECE. Grid search: one parameter, tiny search space,
    and it avoids depending on a solver."""
    best_t, best_e = 1.0, float("inf")
    t = lo
    while t <= hi + 1e-9:
        e = ece(pairs_from_rows(rows, T=t))
        if e < best_e:
            best_t, best_e = round(t, 2), e
        t += step
    return best_t, best_e


def fit_smoothing(rows: list[dict], lo=0.0, hi=0.6, step=0.01) -> tuple[float, float]:
    best_e_, best_eps = float("inf"), 0.0
    e_ = lo
    while e_ <= hi + 1e-9:
        v = ece(pairs_from_rows(rows, eps=e_))
        if v < best_e_:
            best_eps, best_e_ = round(e_, 2), v
        e_ += step
    return best_eps, best_e_


def load_table(path: Path | str | None = None) -> dict:
    p = Path(path or DEFAULT_FILE)
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("models", {})


def temperature_for(model: str, path: Path | str | None = None) -> float:
    """Fitted T for a model name, or 1.0 if it has never been calibrated."""
    entry = load_table(path).get(model)
    return float(entry["temperature"]) if entry else 1.0
