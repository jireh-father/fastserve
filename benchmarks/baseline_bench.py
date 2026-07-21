"""Naive-baseline-only pass: HF eager bf16, unquantized, no speculative
decoding — the same reference point fastserve's own `bench.py
--compare-baseline` and the original inference-opt campaign used. Run as
its own subprocess (separate CUDA context from the optimized vLLM pass —
vLLM doesn't release its ~68GB reservation until the process exits, so this
can't share a process with run_bench.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastserve.bench import bench_hf_baseline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id to actually load (mirror already resolved)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    result = bench_hf_baseline(args.model, n_prompts=args.n, max_new_tokens=args.max_new)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
