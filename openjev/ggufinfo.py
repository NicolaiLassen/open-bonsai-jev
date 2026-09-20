#!/usr/bin/env python3
"""
Read a GGUF file's header to find out what it actually is.

This replaces guessing from the filename. A renamed `.gguf` used to be enough
to fool the old check into loading a Bonsai 2 file on a runtime that cannot
execute it; the header cannot be renamed.

Only the header is read -- a few hundred KB, not the weights.

Usage:
    ggufinfo.py model.gguf [more.gguf ...]
"""

from __future__ import annotations

import collections
import struct
import sys
from pathlib import Path

# ggml tensor type ids we care about. Everything else is reported numerically.
#
# The Prism ids were read off the published files rather than taken from a
# header, so treat them as observed rather than promised.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    41: "Q1_0",     # Bonsai 1, 1-bit. Merged upstream.
    42: "Q2_0",     # ternary, 2-bit slots. Two incompatible layouts exist.
    142: "PQ2_0",   # Bonsai 2 ternary, 2-bit slots. Prism only.
    143: "PTQ1_0",  # Bonsai 2 ternary, dense trit packing. Prism only.
}

# Types stock llama.cpp does not implement at all. These are *tensor* type ids,
# read from the tensor table -- not `general.file_type`, which is a different
# enum and does not line up (a PQ2_0 file reports file_type 141, tensors 142).
PRISM_ONLY = {142, 143}


class GGUFError(Exception):
    pass


def read_gguf_info(path) -> dict:
    """Return {'version', 'file_type', 'tensor_types', 'dominant', 'name'}."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError(f"{path.name} is not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise GGUFError(f"{path.name}: unsupported GGUF version {version}")
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def rd_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_val(t):
            if t == 8:                                   # string
                n, = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
            elif t == 9:                                 # array
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                if et in fixed:                          # fast path
                    f.seek(fixed[et] * n, 1)
                else:
                    for _ in range(n):
                        skip_val(et)
            elif t in fixed:
                f.seek(fixed[t], 1)
            else:
                raise GGUFError(f"{path.name}: unknown metadata value type {t}")

        file_type = None
        for _ in range(n_kv):
            key = rd_str()
            t, = struct.unpack("<I", f.read(4))
            if key == "general.file_type" and t in (4, 5):
                file_type, = struct.unpack("<I", f.read(4))
            else:
                skip_val(t)

        counts = collections.Counter()
        for _ in range(n_tensors):
            rd_str()
            nd, = struct.unpack("<I", f.read(4))
            f.seek(8 * nd, 1)
            tt, = struct.unpack("<I", f.read(4))
            f.seek(8, 1)
            counts[tt] += 1

    # The dominant *quantized* type is what decides runtime support; every file
    # also carries F32 norms and such.
    quantized = {t: n for t, n in counts.items() if t not in (0, 1, 30)}
    dominant = max(quantized, key=quantized.get) if quantized else (
        max(counts, key=counts.get) if counts else None
    )
    return {
        "path": path,
        "version": version,
        "file_type": file_type,
        "tensor_types": dict(counts),
        "dominant": dominant,
        "name": type_name(dominant),
    }


def type_name(type_id) -> str:
    if type_id is None:
        return "unknown"
    return GGML_TYPES.get(type_id, f"type{type_id}")


def needs_prism_runtime(info: dict) -> bool:
    """True if any tensor uses a type only PrismML's llama.cpp implements."""
    return any(t in PRISM_ONLY for t in info["tensor_types"])


def main(argv=None):
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__.strip())
        return 2
    for p in args:
        try:
            i = read_gguf_info(p)
        except (GGUFError, OSError) as e:
            print(f"{Path(p).name:44s} ERROR: {e}")
            continue
        top = ", ".join(
            f"{type_name(t)}x{n}"
            for t, n in sorted(i["tensor_types"].items(), key=lambda kv: -kv[1])[:3]
        )
        flag = "  [needs PrismML fork]" if needs_prism_runtime(i) else ""
        print(f"{Path(p).name:44s} {i['name']:8s} gguf v{i['version']}  {top}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
