"""Run compare3 (original vs vLLM vs fastserve) over every published model,
pruning the HF cache between so a 12-model sweep doesn't fill disk.
"""
import json
import os
import shutil
import subprocess
import sys
import time

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
HF_HOME = os.path.join(ROOT, "..", ".hf_cache")
load_dotenv(os.path.join(ROOT, ".env"))

BASES = [
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-2b-it", "google/gemma-2-9b-it",
    "01-ai/Yi-1.5-9B-Chat", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]


def cache_dir(model_id):
    return os.path.join(HF_HOME, "hub", "models--" + model_id.replace("/", "--"))


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    t0 = time.time()
    for i, base in enumerate(BASES):
        short = base.split("/")[-1]
        out = os.path.join(HERE, "results", f"cmp_{short}.json")
        log = os.path.join(HERE, "results", f"cmp_{short}.log")
        print(f"\n[{i+1}/{len(BASES)}] {base} (+{round(time.time()-t0)}s)", flush=True)
        with open(log, "w") as f:
            subprocess.run([VENV_PY, os.path.join(HERE, "compare3.py"),
                            "--base", base, "--n", "8", "--out", out],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("FASTSERVE_GPU", "0"),
                                    HF_HOME=HF_HOME,
                                    VLLM_CACHE_ROOT=os.path.join(ROOT, "..", ".vllm_cache")),
                           stdout=f, stderr=subprocess.STDOUT)
        if os.path.exists(out):
            d = json.load(open(out))
            c = d["configs"]
            print(f"  orig acc={c['original'].get('acc')} {c['original'].get('tok_s')}tok/s | "
                  f"vllm acc={c['vllm'].get('acc')} {c['vllm'].get('tok_s')}tok/s | "
                  f"fastserve acc={c['fastserve'].get('acc')} {c['fastserve'].get('tok_s')}tok/s", flush=True)
        # prune both the bf16 base and the AWQ quant cache
        shutil.rmtree(cache_dir(base), ignore_errors=True)
        det = json.load(open(out)).get("detected_quant") if os.path.exists(out) else None
        if det:
            shutil.rmtree(cache_dir(det), ignore_errors=True)
    print(f"\n=== compare batch done ({round(time.time()-t0)}s) ===", flush=True)


if __name__ == "__main__":
    main()
