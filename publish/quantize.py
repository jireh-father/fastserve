"""Self-quantization pipeline orchestrator: quantize a model with AutoAWQ,
gate it against its own bf16 baseline accuracy, publish to the Hub only if
it passes. Runs the two stages as separate subprocesses (clean CUDA context
each — AutoAWQ/plain-HF vs vLLM don't mix well in one process, same reason
every other vLLM stage in this project runs isolated).

Usage:
    fastserve/.venv/bin/python publish/quantize.py --model Qwen/Qwen3.5-4B
    fastserve/.venv/bin/python publish/quantize.py --model Qwen/Qwen3.5-4B --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
FASTSERVE_ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(FASTSERVE_ROOT, ".venv", "bin", "python")
# Same NFS-backed cache dirs the benchmark driver uses — /home/work is a 49GB
# loop partition that fills up fast (see benchmarks/driver.py).
HF_HOME = os.path.join(FASTSERVE_ROOT, "..", ".hf_cache")
VLLM_CACHE_ROOT = os.path.join(FASTSERVE_ROOT, "..", ".vllm_cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None, help="defaults to publish/artifacts/<model-short-name>")
    ap.add_argument("--namespace", default=os.environ.get("FASTSERVE_HF_NAMESPACE"),
                     help="HF namespace to publish under (env: FASTSERVE_HF_NAMESPACE; "
                          "defaults to the HF_TOKEN account itself)")
    ap.add_argument("--repo-name", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-drop", type=float, default=0.10)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="quantize + gate, skip the actual upload")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES — check it's idle first")
    ap.add_argument("--calib-dataset", default="ultrachat-200k",
                     help="wikitext (+ --calib-config wikitext-2-raw-v1) for chat templates that "
                          "reject system-role messages, e.g. Gemma")
    ap.add_argument("--calib-config", default=None)
    ap.add_argument("--method", choices=["awq", "w8a8"], default="awq",
                     help="awq = W4A16 (default); w8a8 = INT8 (A100-friendly)")
    args = ap.parse_args()

    load_dotenv(os.path.join(FASTSERVE_ROOT, ".env"))

    suffix = "W8A8-INT8" if args.method == "w8a8" else "AWQ"
    out_dir = args.out_dir or os.path.join(HERE, "artifacts", args.model.split("/")[-1] + "-" + suffix)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HOME=HF_HOME, VLLM_CACHE_ROOT=VLLM_CACHE_ROOT)

    print(f"=== stage 1/2: quantize {args.model} -> {out_dir} ===", flush=True)
    stage1 = [
        VENV_PY, os.path.join(HERE, "quantize_local.py"),
        "--model", args.model, "--out-dir", out_dir, "--n", str(args.n),
        "--calib-dataset", args.calib_dataset, "--method", args.method,
    ]
    if args.calib_config:
        stage1 += ["--calib-config", args.calib_config]
    rc = subprocess.run(stage1, env=env).returncode
    if rc != 0:
        print(f"stage 1 failed (rc={rc})", file=sys.stderr)
        sys.exit(rc)

    print(f"\n=== stage 2/2: validate + publish ===", flush=True)
    cmd = [
        VENV_PY, os.path.join(HERE, "validate_and_publish.py"),
        "--local-dir", out_dir,
        "--n", str(args.n), "--max-drop", str(args.max_drop),
    ]
    if args.namespace:
        cmd += ["--namespace", args.namespace]
    if args.repo_name:
        cmd += ["--repo-name", args.repo_name]
    if args.private:
        cmd.append("--private")
    if args.dry_run:
        cmd.append("--dry-run")
    rc = subprocess.run(cmd, env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
