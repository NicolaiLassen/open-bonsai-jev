#!/usr/bin/env python3
"""
TypeSafe Jev as a backend, so the hosted System One model can be measured on
exactly the same tasks as the local Bonsai ones.

Jev is the thing this repository reimplements the shape of, so it is the
natural reference point: same eval files, same option lists, same scoring
harness, only the decider changes. It is a hosted API rather than a local
forward pass, so its "latency" includes a network round trip and is not
comparable to a local forward pass in kind, only in what a caller experiences.

Docs: https://docs.typesafe.ai/api. The key is read from TYPESAFE_API_KEY and
is never written anywhere.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


def _key(name: str, used: set[str]) -> str:
    """A criteria key for one option.

    Criteria keys carry meaning to the model -- they are the option names, not
    opaque ids -- so the option text is the key. Only uniqueness and length are
    enforced.
    """
    k = " ".join(str(name).split())[:120] or "option"
    if k in used:
        i = 2
        while f"{k} ({i})" in used:
            i += 1
        k = f"{k} ({i})"
    used.add(k)
    return k


class JevBackend:
    """Same interface as the local backends: one typed question in, one
    distribution over the supplied options out."""

    name = "jev"

    def __init__(self, model: str = MODEL, api_key: str | None = None,
                 timeout: float = 60.0, max_retries: int = 5,
                 verbose: bool = False):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("set TYPESAFE_API_KEY")
        self.input_tokens = 0
        self.calls = 0

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                ENDPOINT, data=data,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                # 429 carries retry-after; 5xx is worth another go.
                if e.code == 429 or 500 <= e.code < 600:
                    wait = float(e.headers.get("retry-after") or 0) or 2 ** attempt
                    if attempt == self.max_retries - 1:
                        raise JevError(f"{e.code} after {self.max_retries} tries") from e
                    time.sleep(min(wait, 30))
                    continue
                body = e.read().decode(errors="replace")[:400]
                raise JevError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise JevError(f"connection failed: {e}") from e
                time.sleep(2 ** attempt)
        raise JevError("unreachable")

    def score_question(self, question: str, options: list[str],
                       state=None) -> dict:
        used: set[str] = set()
        keys = [_key(o, used) for o in options]
        payload = {
            "state": state if state is not None else "",
            "model": self.model,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": question,
                    # Keys are the option text; the model reads them, so no
                    # separate description is needed or wanted here.
                    "criteria": {k: None for k in keys},
                }
            },
        }
        t0 = time.perf_counter()
        out = self._post(payload)
        elapsed = time.perf_counter() - t0

        try:
            ans = out["answers"]["decision"]
            probs = ans["probabilities"]
        except (KeyError, TypeError) as e:
            raise JevError(f"unexpected response: {str(out)[:300]}") from e

        usage = out.get("usage") or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.calls += 1

        # Hand back log-probabilities so the shared softmax over option letters
        # in score.py reproduces these probabilities exactly.
        logits, mass = {}, 0.0
        from score import LETTERS
        for i, k in enumerate(keys):
            p = float(probs.get(k, 0.0))
            mass += p
            logits[LETTERS[i]] = math.log(p) if p > 0 else -1e30
        return {
            "logits": logits,
            "label_mass": mass,
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "forward_s": elapsed,
            "confidence": ans.get("confidence"),
        }

    def score(self, messages, options):
        """Compatibility shim. Jev takes a question and a state, not a chat
        transcript, so a caller that built messages loses nothing by passing
        the last user turn through as the question."""
        question = next((m["content"] for m in reversed(messages)
                         if m.get("role") == "user"), "")
        return self.score_question(question, options, None)

    def close(self):
        pass
