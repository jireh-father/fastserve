"""Batch self-quantize + gate + publish across a list of models. Each model
is handed to quantize.py (which runs its own two isolated subprocesses), then
its local artifact and the source model's HF cache are pruned so a long batch
doesn't fill the disk.

Model list is text-only, accessible (ungated or license already accepted on
this account), non-multimodal — the ones excluded are documented in
SKIPPED below with the reason, so this isn't a silent subset.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
FASTSERVE_ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(FASTSERVE_ROOT, ".venv", "bin", "python")
HF_HOME = os.path.join(FASTSERVE_ROOT, "..", ".hf_cache")
ARTIFACTS = os.path.join(HERE, "artifacts")
LOG_DIR = os.path.join(HERE, "logs")

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-2-2b-it",   # community AWQ was broken — this is the fix
    "google/gemma-2-9b-it",   # community AWQ was broken — this is the fix
    "01-ai/Yi-1.5-9B-Chat",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

# Documented so the batch isn't a silent subset — why each accessible-looking
# model is NOT here.
SKIPPED = {
    "mistralai/Mistral-7B-Instruct-v0.3": "already published (pilot)",
    "Qwen/Qwen2.5-72B-Instruct": "bf16 baseline needs ~144GB > single 80GB GPU (no accuracy gate possible)",
    "Qwen/Qwen3.5-4B": "natively multimodal (vision_config) — text-only re-save breaks vLLM loader",
    "Qwen/Qwen3.6-27B": "natively multimodal (vision_config)",
    "google/gemma-4-12B-it": "natively multimodal (Gemma4UnifiedForConditionalGeneration)",
    "meta-llama/Llama-3.2-3B-Instruct": "gated 403 — license not accepted on this account",
    "meta-llama/Llama-3.1-8B-Instruct": "gated 403 — license not accepted on this account",
    "microsoft/Phi-3.5-mini-instruct": "custom modeling code (DynamicCache.seen_tokens) breaks HF-eager baseline",
    "openai/gpt-oss-20b": "native MXFP4 already; llm-compressor AWQ path not validated for it",
    "zai-org/GLM-4.7-Flash": "vLLM AWQ+MLA-attention bug (would fail the gate's vLLM load)",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "custom tokenizer incompatible with current transformers",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "gated + transformers config mismatch + huge MoE",
    "allenai/tmax-sft-8b": "agentic/tool-use model — GSM8K gate misreads it, not a fair accuracy signal",
}


def hf_cache_dir_for(model_id: str) -> str:
    return os.path.join(HF_HOME, "hub", "models--" + model_id.replace("/", "--"))


def main() -> None:
    load_dotenv(os.path.join(FASTSERVE_ROOT, ".env"))
    os.makedirs(LOG_DIR, exist_ok=True)
    gpu = os.environ.get("FASTSERVE_GPU", "0")

    print(f"batch: {len(MODELS)} models, gpu={gpu}", flush=True)
    print(f"(skipping {len(SKIPPED)} others — see SKIPPED in this script)", flush=True)
    results = []
    t0 = time.time()

    for i, model in enumerate(MODELS):
        short = model.split("/")[-1]
        log_path = os.path.join(LOG_DIR, f"batch_{short}.log")
        print(f"\n[{i+1}/{len(MODELS)}] {model} (+{round(time.time()-t0)}s) -> {log_path}", flush=True)

        cmd = [VENV_PY, os.path.join(HERE, "quantize.py"), "--model", model, "--n", "30", "--gpu", gpu]
        with open(log_path, "w") as logf:
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode

        published = None
        if rc == 0:
            for line in reversed(open(log_path).read().splitlines()):
                if line.startswith("PUBLISHED:"):
                    published = line.split("PUBLISHED:", 1)[1].strip()
                    break
        status = "OK" if (rc == 0 and published) else f"FAIL(rc={rc})"
        results.append({"model": model, "status": status, "published": published})
        print(f"  {status} {published or ''}", flush=True)

        # Reclaim disk: drop the local quant artifact (it's on the Hub now) and
        # the source model's bf16 cache before the next model.
        art = os.path.join(ARTIFACTS, short + "-AWQ")
        if published and os.path.isdir(art):
            shutil.rmtree(art, ignore_errors=True)
        shutil.rmtree(hf_cache_dir_for(model), ignore_errors=True)

    print(f"\n=== batch done ({round(time.time()-t0)}s) ===", flush=True)
    ok = [r for r in results if r["status"] == "OK"]
    print(f"published {len(ok)}/{len(MODELS)}:", flush=True)
    for r in results:
        print(f"  {r['status']:>12}  {r['model']}  {r['published'] or ''}", flush=True)
    with open(os.path.join(HERE, "batch_results.json"), "w") as f:
        json.dump({"results": results, "skipped": SKIPPED}, f, indent=2)


if __name__ == "__main__":
    main()
