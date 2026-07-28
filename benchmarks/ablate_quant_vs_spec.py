"""Ablation: separate what quantization contributes from what fastserve's
speculative decoding adds on top.

The 3-way table compares out-of-the-box setups (plain vLLM on the *original*
bf16 weights vs fastserve on its auto-detected quant + spec), so the gap between
those two columns mixes both effects. This measures the missing 4th cell —
vLLM serving the *same quantized checkpoint* with speculative decoding off — so
each contribution can be read separately:

    vllm_bf16  ->  vllm_quant   = quantization alone
    vllm_quant ->  fastserve    = speculative decoding alone

Writes results/ablate_<short>.json and prints a summary table.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))
GPU = os.environ.get("FASTSERVE_GPU", "1")

# Representative spread: small/large, AWQ/W8A8, ngram/eagle3 spec.
MODELS = os.environ.get("ABLATE_MODELS", ",".join([
    "upstage/SOLAR-10.7B-Instruct-v1.0",   # AWQ  + ngram
    "google/gemma-4-12B-it",               # W8A8 + eagle3
    "google/gemma-4-26B-A4B-it",           # W8A8 + eagle3, biggest reported win
])).split(",")


def worker(model, spec, n, mem_util=0.85):
    cmd = [VENV_PY, os.path.join(HERE, "compare3_worker.py"),
           "--mode", "vllm", "--model", model, "--n", str(n)]
    if spec:
        cmd += ["--spec", spec]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
               HF_HOME=os.path.join(ROOT, "..", ".hf_cache"),
               VLLM_CACHE_ROOT=os.path.join(ROOT, "..", ".vllm_cache"),
               FASTSERVE_GPU_MEM_UTIL=str(mem_util))
    env.pop("FASTSERVE_ENFORCE_EAGER", None)
    # Stream to a per-model log instead of capturing: a captured pipe hides all
    # progress, so a stalled download or a wedged engine looks identical to work
    # in progress (one run sat in poll() for 17h before this was noticed). The
    # timeout turns a hang into a recorded failure instead of an infinite wait.
    log = os.path.join(HERE, "logs", f"ablate_{model.split('/')[-1]}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    timeout_s = int(os.environ.get("ABLATE_TIMEOUT", "3600"))
    with open(log, "w") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "tok_s": None}
    txt = open(log).read()
    m = re.search(r"^RESULT (\{.*\})$", txt, re.MULTILINE)
    if not m:
        return {"error": True, "tok_s": None, "tail": txt[-400:]}
    return json.loads(m.group(1))


def main():
    from fastserve.detect import detect
    n = int(os.environ.get("BENCH_N", "8"))
    rows = []
    for base in MODELS:
        short = base.split("/")[-1]
        det = detect(base)
        quant = det.quantized_model
        spec = det.eagle_model or "ngram"
        print(f"\n=== {base}  quant={quant}  spec={spec} ===", flush=True)

        # the missing cell: same quantized weights, speculative decoding OFF
        print("[quant, no spec] ...", flush=True)
        r = worker(quant, None, n)
        print("   ", {k: r.get(k) for k in ("acc", "tok_s")}, flush=True)

        prev = {}
        p = os.path.join(RESULTS, f"cmp_{short}.json")
        if os.path.exists(p):
            prev = json.load(open(p)).get("configs", {})
        row = {
            "base": base, "quant": quant, "spec": spec,
            "vllm_bf16": prev.get("vllm", {}).get("tok_s"),
            "vllm_quant_nospec": r.get("tok_s"),
            "fastserve": prev.get("fastserve", {}).get("tok_s"),
        }
        rows.append(row)
        json.dump(row, open(os.path.join(RESULTS, f"ablate_{short}.json"), "w"), indent=2)

    print("\n=== ablation summary (tok/s) ===", flush=True)
    print(f"{'model':<28}{'vLLM bf16':>11}{'vLLM quant':>12}{'fastserve':>11}"
          f"{'quant x':>9}{'spec x':>8}", flush=True)
    for r in rows:
        b, q, f = r["vllm_bf16"], r["vllm_quant_nospec"], r["fastserve"]
        qx = f"{q/b:.2f}x" if (b and q) else "-"
        sx = f"{f/q:.2f}x" if (q and f) else "-"
        print(f"{r['base'].split('/')[-1]:<28}{str(b):>11}{str(q):>12}{str(f):>11}"
              f"{qx:>9}{sx:>8}", flush=True)
    print("\n=== ablation done ===", flush=True)


if __name__ == "__main__":
    main()
