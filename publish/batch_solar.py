"""Publish run for the two Upstage Solar models. Both are text-only and
vLLM-native (LlamaForCausalLM / SolarForCausalLM), so they take the standard
AWQ path — smaller model first so a pipeline problem surfaces cheaply.

solar-pro-preview is a depth-up-scaled model whose config carries block-skip
connection indices (bskcn_*) and ships custom modeling code; llm-compressor
quantizes through transformers so the custom class loads fine, and the skip
wiring lives in the config (not in weights), so it survives quantization.
"""
import json
import os
import shutil
import subprocess
import time

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
HF_HOME = os.path.join(ROOT, "..", ".hf_cache")
load_dotenv(os.path.join(ROOT, ".env"))
GPU = os.environ.get("FASTSERVE_GPU", "1")

# (model, method, calib_dataset, calib_config)
MODELS = [
    ("upstage/SOLAR-10.7B-Instruct-v1.0", "awq", "ultrachat-200k", None),
    ("upstage/solar-pro-preview-instruct", "awq", "ultrachat-200k", None),
]


def cache_dir(model_id):
    return os.path.join(HF_HOME, "hub", "models--" + model_id.replace("/", "--"))


def already_published(api, ns, short):
    from huggingface_hub.utils import RepositoryNotFoundError
    for suf in ("AWQ", "W8A8-INT8"):
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
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    ns = os.environ.get("FASTSERVE_HF_NAMESPACE") or api.whoami()["name"]
    print(f"publish namespace: {ns}", flush=True)
    results = []
    t0 = time.time()
    for i, (model, method, calib, cfg) in enumerate(MODELS):
        short = model.split("/")[-1]
        log = os.path.join(HERE, "logs", f"solar_{short}.log")
        exists = already_published(api, ns, short)
        if exists:
            print(f"\n[{i+1}/{len(MODELS)}] {model} — already published ({exists}), skip", flush=True)
            results.append({"model": model, "status": "SKIP",
                            "published": f"https://huggingface.co/{exists}"})
            continue
        print(f"\n[{i+1}/{len(MODELS)}] {model} ({method}, {calib}) (+{round(time.time()-t0)}s)", flush=True)
        cmd = [VENV_PY, os.path.join(HERE, "quantize.py"), "--model", model,
               "--method", method, "--gpu", GPU, "--n", "15", "--calib-dataset", calib]
        if cfg:
            cmd += ["--calib-config", cfg]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        txt = open(log).read()
        published = None
        for line in reversed(txt.splitlines()):
            if line.startswith("PUBLISHED:"):
                published = line.split("PUBLISHED:", 1)[1].strip()
                break
        status = "OK" if published else ("GATE_FAIL" if "GATE FAILED" in txt else f"FAIL(rc={rc})")
        results.append({"model": model, "status": status, "published": published})
        print(f"  {status}  {published or ''}", flush=True)
        suffix = "W8A8-INT8" if method == "w8a8" else "AWQ"
        art = os.path.join(HERE, "artifacts", short + "-" + suffix)
        if published and os.path.isdir(art):
            shutil.rmtree(art, ignore_errors=True)
        shutil.rmtree(cache_dir(model), ignore_errors=True)

    print(f"\n=== solar batch done ({round(time.time()-t0)}s) ===", flush=True)
    for r in results:
        print(f"  {r['status']:>10}  {r['model']}  {r['published'] or ''}", flush=True)
    json.dump(results, open(os.path.join(HERE, "batch_solar_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
