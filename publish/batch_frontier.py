"""Ordered publish run for the frontier models the user asked for, easiest
first (small dense → large dense → MoE). Each model goes through quantize.py
(quantize → accuracy gate → upload to glenic/); failures are logged and the
driver moves on — never stops. Cache + artifact pruned between models.

Infeasible on this hardware (documented, not attempted here):
  - Kimi-Linear-48B-A3B : custom tokenizer import error + 256-expert MoE + linear attn
  - Leanstral-1.5-119B  : bf16 ~238 GiB can't be loaded to quantize
  - gpt-oss-20b / 120b  : already ship native MXFP4 (re-quantizing is pointless)
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
GPU = os.environ.get("FASTSERVE_GPU", "1")  # GPU1 free; GPU0 has a foreign job

# (model, method, calib_dataset, calib_config).
# Gemma-4 are multimodal *ForConditionalGeneration wrappers not in llm-compressor's
# AWQ mapping registry → AWQ's generic mappings can't segment the nested
# `language_model.layers.*` tree and error out. W8A8 RTN needs no mappings and, on
# A100 (INT8 tensor cores, no FP8), is the faster format anyway — so Gemma goes
# W8A8 directly. Qwen3.x is handled by llm-compressor's *dynamic* AWQ registry
# (hybrid-attention aware), so it keeps 4-bit AWQ; the pipeline auto-falls-back to
# W8A8 if that ever errors.
MODELS = [
    ("google/gemma-4-E2B-it", "w8a8", "wikitext", "wikitext-2-raw-v1"),
    ("google/gemma-4-E4B-it", "w8a8", "wikitext", "wikitext-2-raw-v1"),
    ("google/gemma-4-12B-it", "w8a8", "wikitext", "wikitext-2-raw-v1"),
    ("Qwen/Qwen3.6-27B", "awq", "ultrachat-200k", None),
    ("google/gemma-4-31B-it", "w8a8", "wikitext", "wikitext-2-raw-v1"),
    ("google/gemma-4-26B-A4B-it", "w8a8", "wikitext", "wikitext-2-raw-v1"),
]


def cache_dir(model_id):
    return os.path.join(HF_HOME, "hub", "models--" + model_id.replace("/", "--"))


def already_published(api, ns, short):
    """Idempotency: skip a model whose quant repo already exists (either suffix,
    since an AWQ job can auto-fall-back to W8A8)."""
    from huggingface_hub.utils import RepositoryNotFoundError
    for suf in ("W8A8-INT8", "AWQ"):
        rid = f"{ns}/{short}-{suf}"
        try:
            api.model_info(rid)
            return rid
        except RepositoryNotFoundError:
            continue
        except Exception:
            continue
    return None


def main():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    from huggingface_hub import HfApi
    _tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=_tok)
    ns = os.environ.get("FASTSERVE_HF_NAMESPACE") or api.whoami()["name"]
    print(f"publish namespace: {ns}", flush=True)
    results = []
    t0 = time.time()
    for i, (model, method, calib, cfg) in enumerate(MODELS):
        short = model.split("/")[-1]
        log = os.path.join(HERE, "logs", f"frontier_{short}.log")
        exists = already_published(api, ns, short)
        if exists:
            print(f"\n[{i+1}/{len(MODELS)}] {model} — already published ({exists}), skip", flush=True)
            results.append({"model": model, "method": method, "status": "SKIP", "published": f"https://huggingface.co/{exists}"})
            continue
        print(f"\n[{i+1}/{len(MODELS)}] {model} ({method}, {calib}) (+{round(time.time()-t0)}s)", flush=True)
        cmd = [VENV_PY, os.path.join(HERE, "quantize.py"), "--model", model,
               "--method", method, "--gpu", GPU, "--n", "15", "--calib-dataset", calib]
        if cfg:
            cmd += ["--calib-config", cfg]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        published = None
        txt = open(log).read()
        for line in reversed(txt.splitlines()):
            if line.startswith("PUBLISHED:"):
                published = line.split("PUBLISHED:", 1)[1].strip()
                break
        status = "OK" if published else ("GATE_FAIL" if "GATE FAILED" in txt else f"FAIL(rc={rc})")
        results.append({"model": model, "method": method, "status": status, "published": published})
        print(f"  {status}  {published or ''}", flush=True)
        # prune to keep disk sane
        suffix = "W8A8-INT8" if method == "w8a8" else "AWQ"
        art = os.path.join(HERE, "artifacts", short + "-" + suffix)
        if published and os.path.isdir(art):
            shutil.rmtree(art, ignore_errors=True)
        shutil.rmtree(cache_dir(model), ignore_errors=True)

    print(f"\n=== frontier batch done ({round(time.time()-t0)}s) ===", flush=True)
    for r in results:
        print(f"  {r['status']:>10}  {r['model']}  {r['published'] or ''}", flush=True)
    json.dump(results, open(os.path.join(HERE, "batch_frontier_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
