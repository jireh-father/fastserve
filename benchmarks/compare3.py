"""3-way comparison for one model, for the model-card tables: original
(naive HF-eager bf16) vs plain vLLM (bf16) vs fastserve (whatever
`detect()` actually picks — our AWQ + speculative). Reports GSM8K accuracy,
single-stream decode tok/s, and model-weight memory (GiB).

Each config runs as its own subprocess (isolated CUDA context). Memory =
on-disk weight footprint from the HF Hub file sizes (bf16 vs AWQ), which is
the honest cross-config VRAM-for-weights number — vLLM's KV-cache
pre-allocation is a separate, config-independent knob.

  python compare3.py --base Qwen/Qwen3-8B --out results/cmp_Qwen3-8B.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "src"))
ENV = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=os.environ.get("FASTSERVE_GPU", "0"),
    HF_HOME=os.path.join(ROOT, "..", ".hf_cache"),
    VLLM_CACHE_ROOT=os.path.join(ROOT, "..", ".vllm_cache"),
)


def weight_gib(model_id: str) -> float | None:
    """Sum of *.safetensors sizes on the Hub = VRAM needed to hold weights."""
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(model_id, files_metadata=True)
        total = sum(s.size for s in info.siblings
                    if s.rfilename.endswith(".safetensors") and s.size)
        return round(total / 2**30, 2) if total else None
    except Exception:
        return None


def run_worker(mode: str, model: str, n: int, spec: str | None) -> dict:
    cmd = [VENV_PY, os.path.join(HERE, "compare3_worker.py"),
           "--mode", mode, "--model", model, "--n", str(n)]
    if spec:
        cmd += ["--spec", spec]
    proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    m = re.search(r"^RESULT (\{.*\})$", proc.stdout, re.MULTILINE)
    if not m:
        return {"error": True, "tail": (proc.stdout + proc.stderr)[-400:]}
    return json.loads(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from fastserve.detect import detect
    det = detect(args.base)
    quant = det.quantized_model or args.base
    spec = det.eagle_model or "ngram"

    result = {"base": args.base, "detected_quant": det.quantized_model,
              "detected_spec": "eagle3:" + det.eagle_model if det.eagle_model else "ngram",
              "n": args.n, "configs": {}}

    print(f"[1/3] original HF-eager bf16: {args.base}", flush=True)
    r = run_worker("hf", args.base, args.n, None)
    r["weight_gib"] = weight_gib(args.base)
    result["configs"]["original"] = r
    print("     ", r, flush=True)

    print(f"[2/3] plain vLLM bf16: {args.base}", flush=True)
    r = run_worker("vllm", args.base, args.n, None)
    r["weight_gib"] = weight_gib(args.base)
    result["configs"]["vllm"] = r
    print("     ", r, flush=True)

    print(f"[3/3] fastserve: {quant} + {spec}", flush=True)
    r = run_worker("vllm", quant, args.n, spec)
    r["weight_gib"] = weight_gib(quant)
    result["configs"]["fastserve"] = r
    print("     ", r, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
