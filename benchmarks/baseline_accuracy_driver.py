"""Runs baseline_accuracy.py (naive bf16 GSM8K, n=30) for every model where
the quantized side already has a measured GSM8K number and a baseline is
actually feasible (excludes Qwen2.5-72B: bf16 doesn't fit; excludes models
whose "optimized" stack never used quantization in the first place —
gpt-oss-20b, tmax-sft-8b — since there's nothing to compare there).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver import ENV, VENV_PY, gpu_mem_used, run_subprocess, wait_for_idle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")
TIMEOUT_S = 3600

# (original_model_id, baseline_model_id_to_actually_load)
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct"),
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    ("meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("google/gemma-2-2b-it", "unsloth/gemma-2-2b-it"),
    ("google/gemma-2-9b-it", "unsloth/gemma-2-9b-it"),
    ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct"),
    ("01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B-Chat"),
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
    ("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B"),
    ("google/gemma-4-12B-it", "google/gemma-4-12B-it"),
]


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = time.time()
    for i, (orig, baseline_model) in enumerate(MODELS):
        idle_mem = wait_for_idle("0")
        print(f"[{i+1}/{len(MODELS)}] {orig} (baseline={baseline_model}) "
              f"(+{round(time.time()-t0)}s elapsed, gpu0={idle_mem}MiB)", flush=True)
        out_json = os.path.join(RESULTS_DIR, safe_name(orig) + ".baseline_acc.json")
        log_path = os.path.join(LOG_DIR, safe_name(orig) + ".baseline_acc.log")
        cmd = [VENV_PY, os.path.join(HERE, "baseline_accuracy.py"),
               "--model", baseline_model, "--long-thinker-name", orig,
               "--out", out_json, "--n", "30"]
        rc = run_subprocess(cmd, log_path, TIMEOUT_S)
        post_mem = gpu_mem_used("0")
        if rc == 0 and os.path.exists(out_json):
            with open(out_json) as f:
                data = json.load(f)
            print(f"  OK acc={data['acc']} trunc={data['truncated']}/{data['n']} "
                  f"wall={data['wall_s']}s post_mem={post_mem}MiB", flush=True)
        else:
            tail = "\n".join(open(log_path).read().splitlines()[-10:])
            print(f"  FAIL(rc={rc}): {tail[-500:]}", flush=True)
    print(f"\nDONE — {len(MODELS)} models, {round(time.time()-t0)}s total", flush=True)


if __name__ == "__main__":
    main()
