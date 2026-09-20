#!/usr/bin/env python3
"""
Semantic-if scoring: ask a typed question, read the answer straight from the
model's next-token scores.

A re-implementation of the "direct" mode described by SemIf (formerly OpenJev)
by TheoLeeCJ -- https://github.com/TheoLeeCJ/SemIf. None of SemIf's code is
used here; the prompt is our own, so numbers are not comparable to theirs.

Two backends:

  torch      a HuggingFace checkpoint, one forward pass, read logits[-1]
  llama.cpp  a .gguf through llama-server's /completion with n_predict=1

Both do the same thing: build a chat prompt whose assistant turn starts exactly
where the answer letter belongs, look up the score of each option letter, and
softmax over just those letters.

Usage:
    score.py -q "Is the sky blue?"
    score.py -q "Is the user angry?" --state '{"msg": "this is broken AGAIN"}'
    score.py -q "What is the priority?" --options low,medium,high
    score.py --input examples/easy100.jsonl --model ../bonzi/models/Bonsai-8B --gpu
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import socket
import string
import subprocess
import tempfile
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Model replies can contain emoji; Windows consoles default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_MODEL = HERE.parent / "bonzi" / "models" / "Ternary-Bonsai-2-27B"
DEFAULT_OPTIONS = ["true", "false"]
LETTERS = string.ascii_uppercase

SYSTEM_PROMPT = (
    "You are a decision function. You are given a state and a question with a "
    "fixed set of options. Choose the single option that best answers the "
    "question. Reply with the option letter only -- no punctuation, no "
    "explanation, no other text."
)


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def build_messages(question: str, options: list[str], state=None) -> list[dict]:
    """The prompt. Options are lettered A., B., ... and the model answers with
    one letter, which is why a single next-token distribution is enough."""
    parts = []
    if state is not None and state != "":
        if isinstance(state, (dict, list)):
            rendered = json.dumps(state, indent=2, ensure_ascii=False)
        else:
            rendered = str(state)
        parts.append(f"State:\n{rendered}")
    parts.append(f"Question: {question}")
    lettered = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    parts.append(f"Options:\n{lettered}")
    parts.append("Answer with the option letter only.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def softmax_over(scores: dict[str, float]) -> dict[str, float]:
    """Softmax restricted to the option letters. Everything else in the
    vocabulary is discarded before this point."""
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exp.values()) or 1.0
    return {k: v / total for k, v in exp.items()}


# --------------------------------------------------------------------------
# torch backend
# --------------------------------------------------------------------------

class TorchBackend:
    """HuggingFace checkpoint in memory. One forward pass per question."""

    name = "torch"

    def __init__(self, model_path: Path, device: str = "auto", threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if device == "cpu" and threads:
            torch.set_num_threads(threads)

        dtype = torch.bfloat16 if device != "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, low_cpu_mem_usage=True
        )
        self.model.to(device)
        self.model.eval()
        self._letter_ids: dict[str, int] = {}

    def letter_id(self, letter: str) -> int:
        """Token id of a bare option letter. Must be a single token -- in every
        vocabulary we have tried, 'A'..'Z' are."""
        if letter not in self._letter_ids:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"option letter {letter!r} is {len(ids)} tokens in this "
                    f"tokenizer, not 1; too many options for direct scoring"
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
        prompt = self.render(messages)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_tokens = int(enc["input_ids"].shape[-1])

        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model(**enc, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        forward_s = time.perf_counter() - t0

        probs = self.torch.softmax(logits, dim=-1)
        raw = {}
        mass = 0.0
        for i, _ in enumerate(options):
            letter = LETTERS[i]
            tid = self.letter_id(letter)
            raw[letter] = float(logits[tid])
            mass += float(probs[tid])
        return {
            "logits": raw,
            "label_mass": mass,
            "prompt_tokens": n_tokens,
            "forward_s": forward_s,
        }

    def close(self):
        pass


# --------------------------------------------------------------------------
# llama.cpp backend
# --------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def find_llama_server(explicit: str | None = None) -> str:
    """llama-server from --llama-server, then the repo-local extract dirs,
    then PATH."""
    if explicit:
        return explicit
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    for d in ("llama.cpp-prism", "llama.cpp-vulkan", "llama.cpp-metal", "llama.cpp"):
        for base in (HERE, HERE.parent / "bonzi", HERE.parent):
            p = base / d / exe
            if p.exists():
                return str(p)
            p = base / d / "build" / "bin" / exe
            if p.exists():
                return str(p)
    from shutil import which
    found = which(exe)
    if found:
        return found
    raise SystemExit(
        "llama-server not found. Run scripts/get_llama_cpp.sh, or pass "
        "--llama-server /path/to/llama-server."
    )


class LlamaCppBackend:
    """A .gguf behind llama-server.

    The prompt is rendered and tokenized *by the server* (/apply-template and
    /tokenize, with --jinja), never by a HuggingFace tokenizer. That is what
    lets one scorer drive any GGUF whatever its vocabulary -- Bonsai 8B is
    Qwen3-based with a 151k vocabulary, Qwen3.5-4B has 248k, and the token ids
    for 'A' differ between them.
    """

    name = "llama.cpp"

    def __init__(
        self,
        model_path: Path,
        gpu: bool = False,
        ctx: int = 4096,
        threads: int | None = None,
        server_url: str | None = None,
        llama_server: str | None = None,
        tokenizer_dir: str | None = None,
        verbose: bool = False,
    ):
        self.proc = None
        self.verbose = verbose
        self.hf_tokenizer = None
        if tokenizer_dir:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        if server_url:
            self.url = server_url.rstrip("/")
            self._wait_ready(timeout=30)
            return

        gguf = resolve_gguf(model_path)
        self.gguf = gguf
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        cmd = [
            find_llama_server(llama_server),
            "-m", str(gguf),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx),
            "--jinja",
            "-ngl", "99" if gpu else "0",
        ]
        if threads:
            cmd += ["-t", str(threads)]
        if not gpu:
            cmd += ["--device", "none"]
        if verbose:
            print(f"[starting {' '.join(cmd)}]", file=sys.stderr)
        # Always keep the server's stderr. A load failure is the most common
        # thing that goes wrong here and its message is the whole diagnosis;
        # throwing it away is what made the old failure mode so opaque.
        self._log = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".llama-server.log", delete=False, encoding="utf-8"
        )
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
        )
        atexit.register(self.close)
        _install_signal_cleanup(self)
        self._wait_ready(timeout=600)

    def _wait_ready(self, timeout: float):
        """llama-server 503s until the weights are in. The first load of a big
        model, or the first Vulkan/Metal run while shaders compile, is slow."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(self._explain_failure())
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception as e:  # noqa: BLE001 - server not up yet
                last = e
            time.sleep(0.5)
        raise SystemExit(f"llama-server did not become ready in {timeout:.0f}s ({last})")

    def server_log(self) -> str:
        log = getattr(self, "_log", None)
        if log is None:
            return ""
        try:
            log.flush()
            return Path(log.name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _explain_failure(self) -> str:
        """Turn a dead llama-server into something actionable."""
        blob = self.server_log()
        errors = [l for l in blob.splitlines() if " E " in l or "error" in l.lower()]
        tail = "\n".join(f"  {l.strip()}" for l in errors[:6]) or "  (no output)"
        msg = [
            f"llama-server exited with code {self.proc.returncode} before "
            f"becoming ready.",
            tail,
        ]
        msg.append(diagnose_load_failure(blob, getattr(self, "gguf", None)))
        return "\n".join(m for m in msg if m)

    def render(self, messages: list[dict]) -> str:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        out = _post(
            f"{self.url}/apply-template",
            {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}},
        )
        return out["prompt"]

    def score(self, messages: list[dict], options: list[str]) -> dict:
        prompt = self.render(messages)
        t0 = time.perf_counter()
        out = _post(
            f"{self.url}/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": 20,
                "temperature": 0.0,
                "cache_prompt": True,
            },
        )
        forward_s = time.perf_counter() - t0

        table = _top_probs(out)
        raw, mass = {}, 0.0
        for i, _ in enumerate(options):
            letter = LETTERS[i]
            p = table.get(letter, 0.0)
            mass += p
            # Work in log space so the shared softmax below matches torch's.
            raw[letter] = math.log(p) if p > 0 else -1e30

        n_tokens = out.get("tokens_evaluated")
        if n_tokens is None:
            n_tokens = len(_post(f"{self.url}/tokenize", {"content": prompt})["tokens"])
        return {
            "logits": raw,
            "label_mass": mass,
            "prompt_tokens": n_tokens,
            "forward_s": forward_s,
        }

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        log = getattr(self, "_log", None)
        if log is not None:
            try:
                log.close()
            except OSError:
                pass


def _install_signal_cleanup(backend) -> None:
    """Stop the server on SIGINT/SIGTERM as well as on a normal exit.

    atexit alone is not enough: it does not run when we are signalled, and an
    orphaned llama-server keeps its weights resident. A 27B left behind holds
    several GB and silently competes with whatever you measure next.
    (Nothing saves us from SIGKILL -- scripts/stop_servers.sh is for that.)
    """
    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def handler(signum, frame, _prev=previous):
            backend.close()
            if callable(_prev):
                _prev(signum, frame)
            else:
                raise SystemExit(128 + signum)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not on the main thread; atexit still applies


def _top_probs(completion: dict) -> dict[str, float]:
    """Pull {token: probability} out of a /completion response.

    llama-server has moved this field around across releases, so accept the
    shapes we have seen. These are a plain softmax over the raw scores and are
    not affected by sampling settings. Anything outside the top n_probs is
    absent, which we treat as probability zero.
    """
    probs = completion.get("completion_probabilities") or []
    if not probs:
        return {}
    first = probs[0]
    entries = first.get("top_logprobs") or first.get("probs") or []
    table = {}
    for e in entries:
        tok = e.get("token")
        if tok is None:
            continue
        if "prob" in e:
            p = float(e["prob"])
        elif "logprob" in e:
            p = math.exp(float(e["logprob"]))
        else:
            continue
        table[tok] = table.get(tok, 0.0) + p
    return table


def diagnose_load_failure(log: str, gguf: Path | None = None) -> str:
    """Map a llama-server load failure onto the thing you actually have to do.

    Every incompatibility we have been able to produce fails *here*, at load,
    within a fraction of a second and with a message. That is why there is no
    pre-flight guess any more: the load itself is the authoritative test.
    """
    low = log.lower()
    hints = []

    if "legacy prism q2_0 layout" in low or (
        "has offset" in low and "expected" in low
    ):
        hints.append(
            "This looks like the legacy Prism Q2_0 layout (group 128). Both "
            "stock llama.cpp and current PrismML builds read Q2_0 as the "
            "official group-64 format and reject it. Use the *_Q2_0_g64.gguf "
            "or the PQ2_0 build of the same model."
        )

    if gguf is not None:
        try:
            from ggufinfo import needs_prism_runtime, read_gguf_info
            info = read_gguf_info(gguf)
            if needs_prism_runtime(info):
                hints.append(
                    f"{gguf.name} uses {info['name']} tensors, which only "
                    f"PrismML's llama.cpp implements. Build it with "
                    f"scripts/build_llama_fork.sh and pass --llama-server, or "
                    f"pick a Q1_0 / Q2_0_g64 model that stock llama.cpp runs."
                )
        except Exception:  # noqa: BLE001 - diagnosis must never itself fail
            pass

    if "unknown model architecture" in low:
        hints.append(
            "The build does not know this architecture -- it is probably older "
            "than the model. Update llama.cpp."
        )
    if "failed to allocate" in low or "out of memory" in low:
        hints.append(
            "Out of memory. Lower --ctx (these models advertise 262K) or drop "
            "GPU offload with --cpu."
        )

    if not hints:
        return ""
    return "\n" + "\n".join(f"hint: {h}" for h in hints)


def resolve_gguf(model_path: Path) -> Path:
    """Accept either a .gguf file or a directory holding exactly one."""
    p = Path(model_path)
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(
            f for f in p.glob("*.gguf") if "mmproj" not in f.name.lower()
        )
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"no .gguf in {p}")
        raise SystemExit(
            f"{p} holds several .gguf files; point --model at one:\n  "
            + "\n  ".join(f.name for f in found)
        )
    raise SystemExit(f"model not found: {p}")


def is_gguf(model_path: Path) -> bool:
    p = Path(model_path)
    if p.is_file():
        return p.suffix == ".gguf"
    return p.is_dir() and any(p.glob("*.gguf"))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def score_with(backend, question, options, state=None) -> dict:
    """Ask one typed question, whichever backend is in use.

    Local backends want a rendered chat prompt; a hosted System One model wants
    the question and the state as separate fields. This is the seam.
    """
    if hasattr(backend, "score_question"):
        return backend.score_question(question, options, state)
    return backend.score(build_messages(question, options, state), options)


def score_one(backend, question, options, state=None, temperature: float = 1.0) -> dict:
    r = score_with(backend, question, options, state)
    # Calibration lives here, one divide before the softmax we already run. It
    # cannot change `choice`: dividing by a positive constant preserves order.
    logits = ({k: v / temperature for k, v in r["logits"].items()}
              if temperature != 1.0 else r["logits"])
    probs_by_letter = softmax_over(logits)
    probs = {options[LETTERS.index(l)]: round(p, 6) for l, p in probs_by_letter.items()}
    choice = max(probs, key=probs.get) if probs else None
    return {
        "choice": choice,
        "probabilities": probs,
        "label_mass": round(r["label_mass"], 6),
        "prompt_tokens": r["prompt_tokens"],
        "forward_s": round(r["forward_s"], 3),
        "backend": backend.name,
        **({"temperature": temperature} if temperature != 1.0 else {}),
    }


def parse_options(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_OPTIONS)
    opts = [o.strip() for o in raw.split(",") if o.strip()]
    if len(opts) < 2:
        raise SystemExit("need at least two options")
    if len(opts) > len(LETTERS):
        raise SystemExit(f"at most {len(LETTERS)} options")
    return opts


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="semantic-if scoring over a local model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    ap.add_argument("-q", "--question")
    ap.add_argument("--state", help="string, or JSON that parses to an object")
    ap.add_argument("--options", help="comma-separated; default true,false")
    ap.add_argument("--input", help="JSONL with question/state/options/expected per row")
    ap.add_argument("--output", help="write JSONL results here")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--backend",
                    choices=["auto", "torch", "llama.cpp", "mlx", "jev"],
                    default="auto")
    ap.add_argument("--gpu", action="store_true", default=True,
                    help="offload to GPU (default)")
    ap.add_argument("--cpu", dest="gpu", action="store_false",
                    help="CPU only")
    ap.add_argument("--device", default="auto", help="torch device: auto|cpu|mps|cuda")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="llama.cpp context; these models advertise 262K and "
                         "will eat your RAM if you let them")
    ap.add_argument("--server", help="use an already-running llama-server URL")
    ap.add_argument("--llama-server", help="path to the llama-server binary")
    ap.add_argument("--tokenizer", help="force HF chat templating for a GGUF model")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="divide option logits by this before the softmax. >1 "
                         "flattens overconfident probabilities and never "
                         "changes the answer. See scripts/calibrate.py")
    ap.add_argument("--calibrated", action="store_true",
                    help="use the fitted temperature for this model from "
                         "results/calibration.json")
    ap.add_argument("--warmup", type=int, default=5,
                    help="questions to run before timing starts (default 5); "
                         "0 disables")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.question and not args.input:
        ap.error("need -q/--question or --input")

    temperature = args.temperature
    if args.calibrated:
        from calibration import temperature_for
        temperature = temperature_for(Path(args.model).name)
        print(f"using fitted temperature {temperature}", file=sys.stderr)

    model_path = Path(args.model)
    backend_kind = args.backend
    if backend_kind == "auto":
        if args.server or is_gguf(model_path):
            backend_kind = "llama.cpp"
        elif (model_path / "runtime" / "vision_artifact.py").exists():
            backend_kind = "mlx"
        else:
            backend_kind = "torch"

    if backend_kind == "jev":
        from jev_backend import JevBackend
        backend = JevBackend(verbose=args.verbose)
    elif backend_kind == "mlx":
        from mlx_backend import MlxBackend
        backend = MlxBackend(model_path, verbose=args.verbose)
    elif backend_kind == "torch":
        backend = TorchBackend(model_path, device=args.device, threads=args.threads)
    else:
        backend = LlamaCppBackend(
            model_path, gpu=args.gpu, ctx=args.ctx, threads=args.threads,
            server_url=args.server, llama_server=args.llama_server,
            tokenizer_dir=args.tokenizer, verbose=args.verbose,
        )

    try:
        if args.question:
            state = _maybe_json(args.state)
            result = score_one(backend, args.question, parse_options(args.options),
                               state, temperature)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        return _run_file(backend, args, temperature)
    finally:
        backend.close()


def _maybe_json(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _run_file(backend, args, temperature: float = 1.0) -> int:
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))

    # Warm up before timing. The first questions through a fresh server pay for
    # shader compilation, paging the weights in, and filling the prompt cache;
    # folding that into the measurement makes small models look far worse than
    # they are, and makes any two runs incomparable.
    n_warmup = min(args.warmup, len(rows))
    if n_warmup:
        print(f"  warming up ({n_warmup} questions, not measured)",
              file=sys.stderr, flush=True)
        for row in rows[:n_warmup]:
            score_one(backend, row["question"],
                      row.get("options") or list(DEFAULT_OPTIONS), row.get("state"),
                      temperature)

    out = open(args.output, "w", encoding="utf-8") if args.output else None
    correct = graded = 0
    t0 = time.perf_counter()
    try:
        for i, row in enumerate(rows, 1):
            options = row.get("options") or list(DEFAULT_OPTIONS)
            result = score_one(backend, row["question"], options, row.get("state"),
                               temperature)
            if "expected" in row:
                graded += 1
                result["expected"] = row["expected"]
                result["correct"] = result["choice"] == row["expected"]
                correct += result["correct"]
            line = json.dumps(result, ensure_ascii=False)
            if out:
                out.write(line + "\n")
                print(f"\r  {i}/{len(rows)}", end="", file=sys.stderr, flush=True)
            else:
                print(line)
    finally:
        if out:
            out.close()
            print(file=sys.stderr)

    elapsed = time.perf_counter() - t0
    rate = len(rows) / elapsed * 60 if elapsed else 0
    summary = (f"scored {len(rows)} questions in {elapsed:.1f} s "
               f"({rate:.0f} questions/min)")
    if graded:
        summary += f", {correct}/{graded} correct"
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
