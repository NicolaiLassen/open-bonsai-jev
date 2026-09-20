#!/usr/bin/env python3
"""
Download model weights, in parallel, without a HuggingFace client dependency.

Two sources:

  hf           huggingface.co. The default; use it if you have plain internet.
  modelscope   modelscope.cn, which mirrors the same repositories and serves
               the binaries itself. Use it where the HuggingFace CDN is
               blocked. It throttles each *connection* to 1-2 MB/s, which is
               why this downloads 128 MiB byte ranges over several connections
               at once -- roughly 9 MB/s against 1-2 for a single stream.

Each segment resumes independently and finished files are checked against the
source's SHA-256 when it publishes one. Dropped connections are normal; a
segment retries up to 5 times.

Usage:
    download_model.py                          # the default model
    download_model.py --list
    download_model.py Bonsai-8B
    download_model.py Ternary-Bonsai-2-27B --source modelscope
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent

SEGMENT = 128 * 1024 * 1024
CONNECTIONS = 8
RETRIES = 5

# name -> (hf repo, modelscope repo, [files], destination folder)
MODELS = {
    # Bonsai 2 -- ternary, needs PrismML's llama.cpp fork
    "Ternary-Bonsai-2-27B": (
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        ["Ternary-Bonsai-2-27B-PTQ1_0.gguf"],
        "bonzi/models/Ternary-Bonsai-2-27B",
    ),
    "Ternary-Bonsai-2-27B-PQ2_0": (
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        ["Ternary-Bonsai-2-27B-PQ2_0.gguf"],
        "bonzi/models/Ternary-Bonsai-2-27B-PQ2_0",
    ),
    # Bonsai 1 -- 1-bit Q1_0, merged upstream, runs on stock llama.cpp
    "Bonsai-8B": (
        "prism-ml/Bonsai-8B-gguf", "prism-ml/Bonsai-8B-gguf",
        ["Bonsai-8B-Q1_0.gguf"], "bonzi/models/Bonsai-8B",
    ),
    "Bonsai-27B": (
        "prism-ml/Bonsai-27B-gguf", "prism-ml/Bonsai-27B-gguf",
        ["Bonsai-27B-Q1_0.gguf"], "bonzi/models/Bonsai-27B",
    ),
    "Bonsai-4B": (
        "prism-ml/Bonsai-4B-gguf", "prism-ml/Bonsai-4B-gguf",
        ["Bonsai-4B-Q1_0.gguf"], "bonzi/models/Bonsai-4B",
    ),
    "Bonsai-1.7B": (
        "prism-ml/Bonsai-1.7B-gguf", "prism-ml/Bonsai-1.7B-gguf",
        ["Bonsai-1.7B-Q1_0.gguf"], "bonzi/models/Bonsai-1.7B",
    ),
    # Qwen reference models, for comparing against a non-Bonsai baseline
    "Qwen3.5-4B-Q4_K_M": (
        "bartowski/Qwen_Qwen3.5-4B-GGUF", "bartowski/Qwen_Qwen3.5-4B-GGUF",
        ["Qwen_Qwen3.5-4B-Q4_K_M.gguf"], "openjev/models/Qwen3.5-4B-Q4_K_M",
    ),
    "Qwen3.5-4B": (
        "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B",
        [
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "openjev/models/Qwen3.5-4B",
    ),
}

DEFAULT = "Ternary-Bonsai-2-27B"


def hf_url(repo: str, f: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{f}"


def ms_url(repo: str, f: str) -> str:
    return f"https://modelscope.cn/models/{repo}/resolve/master/{f}"


def token_header() -> dict:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        p = Path.home() / ".cache" / "huggingface" / "token"
        if p.exists():
            tok = p.read_text().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def remote_info(url: str, headers: dict) -> tuple[int, str | None]:
    """Content-Length and, when the host publishes one, a SHA-256."""
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
        digest = (
            r.headers.get("X-Linked-Etag")
            or r.headers.get("ETag")
            or ""
        ).strip('"')
        # HF ships the LFS sha256 in the etag for big files; 64 hex chars.
        sha = digest if len(digest) == 64 and all(
            c in "0123456789abcdef" for c in digest.lower()
        ) else None
        return size, sha


def fetch_segment(url, headers, start, end, dest: Path, progress) -> None:
    """One byte range into its own .partN file, resumable on its own."""
    part = dest.with_suffix(dest.suffix + f".part{start}")
    have = part.stat().st_size if part.exists() else 0
    want = end - start + 1
    if have >= want:
        progress(have if have == want else 0)
        return

    for attempt in range(RETRIES):
        try:
            h = dict(headers)
            h["Range"] = f"bytes={start + have}-{end}"
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    progress(len(chunk))
            if have >= want:
                return
        except Exception as e:  # noqa: BLE001 - connections drop; that is expected
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))
            have = part.stat().st_size if part.exists() else 0
    raise RuntimeError(f"segment {start}-{end} incomplete")


def download(url: str, dest: Path, headers: dict, connections: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, sha = remote_info(url, headers)
    if dest.exists() and size and dest.stat().st_size == size:
        print(f"  {dest.name}: already complete ({size / 1e9:.2f} GB)")
        return

    label = f"{dest.name} ({size / 1e9:.2f} GB)" if size else dest.name
    print(f"  {label}")

    if not size:  # no Content-Length: single stream, no resume
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return

    ranges = [(s, min(s + SEGMENT, size) - 1) for s in range(0, size, SEGMENT)]
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def progress(n):
        with lock:
            done[0] += n
            pct = done[0] / size * 100
            mbs = done[0] / 1e6 / max(time.time() - t0, 0.001)
            print(f"\r    {pct:5.1f}%  {done[0] / 1e9:5.2f}/{size / 1e9:.2f} GB  "
                  f"{mbs:5.1f} MB/s", end="", flush=True)

    with cf.ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [
            pool.submit(fetch_segment, url, headers, s, e, dest, progress)
            for s, e in ranges
        ]
        for f in cf.as_completed(futures):
            f.result()
    print()

    with open(dest, "wb") as out:
        for s, _ in ranges:
            part = dest.with_suffix(dest.suffix + f".part{s}")
            with open(part, "rb") as p:
                shutil.copyfileobj(p, out)
            part.unlink()

    if sha:
        print("    verifying sha256 ...", end="", flush=True)
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != sha:
            dest.unlink()
            raise SystemExit(f"\n    checksum mismatch for {dest.name}; deleted")
        print(" ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description="download model weights")
    ap.add_argument("model", nargs="?", default=DEFAULT)
    ap.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    ap.add_argument("--connections", type=int, default=CONNECTIONS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print("available models:\n")
        for name, (hf, _, files, dest) in MODELS.items():
            mark = " (default)" if name == DEFAULT else ""
            print(f"  {name}{mark}\n      {hf}\n      -> {dest}")
        return 0

    if args.model not in MODELS:
        raise SystemExit(
            f"unknown model {args.model!r}. --list shows the options."
        )

    hf_repo, ms_repo, files, dest_rel = MODELS[args.model]
    repo = hf_repo if args.source == "hf" else ms_repo
    make_url = hf_url if args.source == "hf" else ms_url
    headers = token_header() if args.source == "hf" else {}
    dest = ROOT / dest_rel

    print(f"{args.model} from {args.source}:{repo}")
    print(f"into {dest}")
    for f in files:
        download(make_url(repo, f), dest / Path(f).name, headers, args.connections)
    print(f"\ndone: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
