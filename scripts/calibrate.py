#!/usr/bin/env python3
"""
Fit a calibration temperature per model from already-scored eval files.

Reports honest numbers: the temperature is fitted on half the rows and the
error is reported on the other half, because fitting and scoring on the same
data flatters the method. Writes results/calibration.json, which
`score.py --calibrated` reads.

    calibrate.py                       # every *.wanli256.jsonl in results/
    calibrate.py --eval easy100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "openjev"))

from calibration import (  # noqa: E402
    ece, fit_smoothing, fit_temperature, pairs_from_rows, reliability,
)

# scored-file stem -> the model name score.py --calibrated will look up
NAMES = {
    "Bonsai-1.7B-Q1_0": "Bonsai-1.7B",
    "Bonsai-4B-Q1_0": "Bonsai-4B",
    "Bonsai-8B-Q1_0": "Bonsai-8B",
    "Bonsai-27B-Q1_0": "Bonsai-27B",
    "Ternary-Bonsai-8B-PQ2_0": "Ternary-Bonsai-8B",
    "Ternary-Bonsai-2-27B-PTQ1_0": "Ternary-Bonsai-2-27B",
    "jev": "jev",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit calibration temperatures")
    ap.add_argument("--eval", default="wanli256")
    ap.add_argument("--out", default=str(ROOT / "results" / "calibration.json"))
    args = ap.parse_args(argv)

    files = sorted((ROOT / "results").glob(f"*.{args.eval}.jsonl"))
    if not files:
        raise SystemExit(f"no results/*.{args.eval}.jsonl; score something first")

    # A fit is only meaningful over the whole eval. An abandoned run that wrote
    # 66 of 256 rows was being fitted and reported as if it were complete.
    expected = sum(1 for _ in (ROOT / "openjev" / "examples"
                               / f"{args.eval}.jsonl").open())

    out = {"eval": args.eval, "method": "temperature on option logits",
           "protocol": "fitted on odd rows, error reported on even rows",
           "models": {}}

    print(f"{'model':24s} {'acc':>6s} {'ECE raw':>8s} {'ECE cal':>8s} "
          f"{'T*':>5s} {'eps*':>6s} {'ECE eps':>8s}")
    print("-" * 70)
    for f in files:
        stem = f.name.rsplit(f".{args.eval}.jsonl", 1)[0]
        name = NAMES.get(stem, stem)
        rows = [json.loads(l) for l in f.open()]
        rows = [r for r in rows if "expected" in r]
        if not rows:
            continue
        if len(rows) < expected:
            print(f"{name:24s} skipped: {len(rows)} of {expected} rows "
                  f"(incomplete run)")
            continue
        fit_rows = [r for i, r in enumerate(rows) if i % 2]
        test_rows = [r for i, r in enumerate(rows) if not i % 2]

        T, _ = fit_temperature(fit_rows)
        eps, _ = fit_smoothing(fit_rows)
        raw = ece(pairs_from_rows(test_rows))
        cal = ece(pairs_from_rows(test_rows, T=T))
        cal_e = ece(pairs_from_rows(test_rows, eps=eps))
        acc = sum(r["correct"] for r in rows) / len(rows)

        out["models"][name] = {
            "temperature": T, "smoothing": eps,
            "accuracy": round(acc, 4),
            "ece_raw": round(raw, 4), "ece_calibrated": round(cal, 4),
            "ece_smoothed": round(cal_e, 4), "n": len(rows),
            "reliability_raw": reliability(pairs_from_rows(rows)),
            "reliability_calibrated": reliability(pairs_from_rows(rows, T=T)),
        }
        print(f"{name:24s} {acc:6.3f} {raw:8.3f} {cal:8.3f} {T:5.2f} "
              f"{eps:6.2f} {cal_e:8.3f}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("accuracy is unchanged by any of this: a positive divisor cannot "
          "reorder the options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
