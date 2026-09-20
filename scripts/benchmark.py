#!/usr/bin/env python3
"""
Benchmark every Bonsai model we have against every eval set, and write the
numbers the README tables and charts are generated from.

Method, so the numbers mean something:
  * one llama-server per (model, eval), started fresh;
  * a warmup pass that is not measured -- the first questions pay for shader
    compilation, paging weights in and filling the prompt cache;
  * the same prompt and the same eval file for every model;
  * accuracy on label-balanced sets, so chance is 1/n_options, not the
    majority-class rate.

Usage:
    benchmark.py                      # everything found under bonzi/models
    benchmark.py --eval wanli256
    benchmark.py --models Bonsai-8B,Ternary-Bonsai-2-27B
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "openjev"))

from ggufinfo import needs_prism_runtime, read_gguf_info  # noqa: E402

STOCK = ROOT / "bonzi" / "llama.cpp" / "llama-server"
FORK = ROOT / "bonzi" / "llama.cpp-prism" / "build" / "bin" / "llama-server"

# display name -> (folder, generation)
MODELS = [
    ("Bonsai 1 1.7B", "Bonsai-1.7B", 1),
    ("Bonsai 1 4B", "Bonsai-4B", 1),
    ("Bonsai 1 8B", "Bonsai-8B", 1),
    ("Bonsai 1 27B", "Bonsai-27B", 1),
    ("Ternary Bonsai 1 8B", "Ternary-Bonsai-8B", 1),
    ("Bonsai 2 27B", "Ternary-Bonsai-2-27B", 2),
]

EVALS = {
    "easy100": "openjev/examples/easy100.jsonl",
    "wanli256": "openjev/examples/wanli256.jsonl",
}


def find_gguf(folder: Path) -> Path | None:
    files = sorted(
        f for f in folder.glob("*.gguf") if "mmproj" not in f.name.lower()
    )
    return files[0] if files else None


def run_one(gguf: Path, eval_path: Path, server: Path, warmup: int) -> dict:
    out = ROOT / "results" / f"{gguf.stem}.{eval_path.stem}.jsonl"
    out.parent.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "openjev" / "score.py"),
        "--input", str(eval_path),
        "--model", str(gguf),
        "--llama-server", str(server),
        "--warmup", str(warmup),
        "--output", str(out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[-600:]}

    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    graded = [r for r in rows if "correct" in r]
    forwards = [r["forward_s"] for r in rows]
    # The scorer prints the measured (post-warmup) elapsed time; parse it back
    # rather than using wall time, which includes model load.
    measured = None
    for line in proc.stderr.splitlines():
        if "questions in" in line:
            try:
                measured = float(line.split("questions in")[1].split("s")[0])
            except (IndexError, ValueError):
                pass
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in graded),
        "accuracy": (sum(r["correct"] for r in graded) / len(graded)) if graded else None,
        "measured_s": measured,
        "questions_per_min": (len(rows) / measured * 60) if measured else None,
        "median_forward_s": statistics.median(forwards),
        "median_label_mass": statistics.median(r["label_mass"] for r in rows),
        "wall_s": round(wall, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="benchmark the Bonsai lineup")
    ap.add_argument("--eval", help="only this eval set")
    ap.add_argument("--models", help="comma-separated folder names")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
    args = ap.parse_args(argv)

    evals = {args.eval: EVALS[args.eval]} if args.eval else EVALS
    wanted = set(args.models.split(",")) if args.models else None

    results = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "generated": time.strftime("%Y-%m-%d"),
        "models": [],
    }

    for label, folder, gen in MODELS:
        if wanted and folder not in wanted:
            continue
        d = ROOT / "bonzi" / "models" / folder
        gguf = find_gguf(d) if d.is_dir() else None
        if gguf is None:
            print(f"-- {label}: not downloaded, skipping", file=sys.stderr)
            continue
        info = read_gguf_info(gguf)
        server = FORK if needs_prism_runtime(info) else STOCK
        entry = {
            "label": label,
            "folder": folder,
            "generation": gen,
            "file": gguf.name,
            "quant": info["name"],
            "size_gb": round(gguf.stat().st_size / 1e9, 3),
            "runtime": "PrismML fork" if server is FORK else "stock llama.cpp",
            "evals": {},
        }
        for name, rel in evals.items():
            print(f"== {label} / {name}", file=sys.stderr, flush=True)
            r = run_one(gguf, ROOT / rel, server, args.warmup)
            entry["evals"][name] = r
            if "error" in r:
                print(f"   FAILED: {r['error'][:200]}", file=sys.stderr)
            else:
                print(f"   {r['correct']}/{r['n']} "
                      f"({r['accuracy']:.1%}) at {r['questions_per_min']:.0f} q/min",
                      file=sys.stderr)
        results["models"].append(entry)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
