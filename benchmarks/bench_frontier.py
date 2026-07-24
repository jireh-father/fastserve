"""3-way speed benchmark (original HF-eager bf16 / plain vLLM bf16 / fastserve
quant+spec) for the 6 frontier models published to glenic, writing one
results/cmp_<short>.json per model in the exact shape build_comparison_readme.py
reads. Each config runs as its own compare3_worker.py subprocess with a
per-config env so big bf16 models can go eager (no CUDA-graph OOM) while the
smaller quantized checkpoint still captures graphs. A config that fails records
tok_s=null and the run moves on — never blocks the others.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "src"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
GPU = os.environ.get("FASTSERVE_GPU", "1")
RESULTS = os.path.join(HERE, "results")

# base id, and whether it's "big" (bf16 won't leave room for CUDA graphs on one
# 80GB card → run the bf16 config eager to avoid OOM; the quant still uses graphs).
MODELS = [
    ("google/gemma-4-E2B-it", False),
    ("google/gemma-4-E4B-it", False),
    ("google/gemma-4-12B-it", False),
    ("google/gemma-4-26B-A4B-it", True),
    ("Qwen/Qwen3.6-27B", True),
    ("google/gemma-4-31B-it", True),
]


def base_env(mem_util, eager):
    e = dict(os.environ,
             CUDA_VISIBLE_DEVICES=GPU,
             HF_HOME=os.path.join(ROOT, "..", ".hf_cache"),
             VLLM_CACHE_ROOT=os.path.join(ROOT, "..", ".vllm_cache"),
             FASTSERVE_GPU_MEM_UTIL=str(mem_util))
    if eager:
        e["FASTSERVE_ENFORCE_EAGER"] = "1"
    else:
        e.pop("FASTSERVE_ENFORCE_EAGER", None)
    return e


def worker(mode, model, n, spec, env):
    cmd = [VENV_PY, os.path.join(HERE, "compare3_worker.py"),
           "--mode", mode, "--model", model, "--n", str(n)]
    if spec:
        cmd += ["--spec", spec]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    m = re.search(r"^RESULT (\{.*\})$", p.stdout, re.MULTILINE)
    if not m:
        return {"error": True, "tok_s": None, "acc": None, "tail": (p.stdout + p.stderr)[-500:]}
    return json.loads(m.group(1))


def weight_gib(model_id):
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(model_id, files_metadata=True)
        total = sum(s.size for s in info.siblings
                    if s.rfilename.endswith(".safetensors") and s.size)
        return round(total / 2**30, 2) if total else None
    except Exception:
        return None


def main():
    from fastserve.detect import detect
    n = int(os.environ.get("BENCH_N", "8"))
    for base, big in MODELS:
        short = base.split("/")[-1]
        out = os.path.join(RESULTS, f"cmp_{short}.json")
        det = detect(base)
        quant = det.quantized_model or base
        spec = det.eagle_model or "ngram"
        print(f"\n=== {base} (big={big}) quant={quant} spec={spec} ===", flush=True)
        result = {"base": base, "detected_quant": det.quantized_model,
                  "detected_spec": ("eagle3:" + det.eagle_model) if det.eagle_model else "ngram",
                  "n": n, "configs": {}}

        # original: HF-eager bf16 (no vLLM knobs)
        print("[1/3] original HF-eager bf16", flush=True)
        r = worker("hf", base, n, None, base_env(0.85, False))
        r["weight_gib"] = weight_gib(base)
        result["configs"]["original"] = r
        print("     ", {k: r.get(k) for k in ("acc", "tok_s")}, flush=True)

        # plain vLLM bf16: big models go eager at high mem_util to fit weights.
        print("[2/3] plain vLLM bf16", flush=True)
        r = worker("vllm", base, n, None, base_env(0.92 if big else 0.85, big))
        r["weight_gib"] = weight_gib(base)
        result["configs"]["vllm"] = r
        print("     ", {k: r.get(k) for k in ("acc", "tok_s")}, flush=True)

        # fastserve: quant + speculative, CUDA graphs on (never eager).
        print("[3/3] fastserve quant + spec", flush=True)
        r = worker("vllm", quant, n, spec, base_env(0.85, False))
        r["weight_gib"] = weight_gib(quant)
        result["configs"]["fastserve"] = r
        print("     ", {k: r.get(k) for k in ("acc", "tok_s")}, flush=True)

        os.makedirs(RESULTS, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print("WROTE", out, flush=True)
    print("\n=== bench_frontier done ===", flush=True)


if __name__ == "__main__":
    main()
