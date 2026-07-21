"""Does quantization actually preserve accuracy? bench_hf_baseline only ever
measured baseline *speed* — this measures baseline GSM8K accuracy (bf16,
unquantized, same chat-template + boxed instruction as the quantized-side
eval) so it can be compared against the already-measured quantized number.

Naive HF eager, batch=1 sequential (matches the project's established
baseline convention) — n defaults to 30, not 150, purely for wall-clock
budget across ~16 models; still enough to catch gross breakage like the
gemma-2 case, noisier for subtle single-digit-point degradation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_tasks import is_long_thinker, run_hf_eager_gsm8k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 model id to load (mirror already resolved)")
    ap.add_argument("--long-thinker-name", default=None,
                     help="original model id, for the long-thinker token-budget heuristic "
                          "(mirror repo names don't always contain the same marker substrings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    long_thinker = is_long_thinker(args.long_thinker_name or args.model)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    result = run_hf_eager_gsm8k(model, tok, args.n, long_thinker)
    result["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
