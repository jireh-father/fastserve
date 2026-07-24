"""Stage 2 of the self-quantization pipeline: load the locally-quantized
checkpoint through vLLM (separate process/CUDA context from stage 1's plain-
HF AutoAWQ work — same reason every other vLLM stage in this project runs
isolated), re-run GSM8K, and only publish to the Hub if the result is within
tolerance of the bf16 baseline stage 1 recorded. Nothing gets uploaded
without passing this gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, HERE)

# Load .env so HF_TOKEN is available when this stage is run standalone (not only
# when quantize.py forwards its environment).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, "..", ".env"))
except Exception:
    pass

from run_bench import gsm8k_eval  # noqa: E402
from model_card import build_model_card  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--namespace", default=os.environ.get("FASTSERVE_HF_NAMESPACE"),
                     help="HF namespace to publish under (env: FASTSERVE_HF_NAMESPACE; "
                          "defaults to the HF_TOKEN account itself)")
    ap.add_argument("--repo-name", default=None, help="defaults to <model-short-name>-AWQ")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-drop", type=float, default=0.10,
                     help="max acceptable absolute GSM8K accuracy drop vs the bf16 baseline")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="run the gate but skip the actual upload")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    # Publish namespace defaults to the token's own account (never a hardcoded
    # name — the token can only create repos under an account it owns). Resolved
    # here, before the expensive gate, so an auth/namespace problem fails fast.
    namespace = args.namespace
    if not namespace:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN not set and no --namespace given — cannot resolve a "
                  "publish namespace.", file=sys.stderr)
            sys.exit(2)
        from huggingface_hub import HfApi
        namespace = HfApi(token=token).whoami()["name"]
        print(f"publish namespace (from HF_TOKEN account): {namespace}", flush=True)

    with open(os.path.join(args.local_dir, "_fastserve_quant_meta.json")) as f:
        meta = json.load(f)
    baseline_acc = meta["baseline"]["acc"]
    source_model = meta["model"]
    long_thinker = meta["long_thinker"]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.local_dir, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    tok = llm.get_tokenizer()

    result = gsm8k_eval(llm, SamplingParams, tok, args.n, long_thinker)
    quant_acc = result["acc"]
    degenerate_frac = result["degenerate"] / result["n"]
    print(f"baseline_acc={baseline_acc} quant_acc={quant_acc} "
          f"degenerate={result['degenerate']}/{result['n']}", flush=True)

    passed = (quant_acc >= baseline_acc - args.max_drop) and degenerate_frac <= 0.3
    gate = {"baseline_acc": baseline_acc, "quant_acc": quant_acc, "max_drop": args.max_drop,
            "degenerate_frac": round(degenerate_frac, 4), "passed": passed}
    with open(os.path.join(args.local_dir, "_fastserve_gate_result.json"), "w") as f:
        json.dump({"meta": meta, "validation": result, "gate": gate}, f, indent=2)

    if not passed:
        print(f"GATE FAILED — not publishing. {gate}", flush=True)
        sys.exit(1)
    print("GATE PASSED.", flush=True)

    scheme = meta.get("quant_config", {}).get("scheme", "")
    suffix = "W8A8-INT8" if scheme == "W8A8" else "AWQ"
    repo_name = args.repo_name or (source_model.split("/")[-1] + "-" + suffix)
    repo_id = f"{namespace}/{repo_name}"

    if args.dry_run:
        print(f"(--dry-run: would publish to {repo_id}, skipping actual upload)", flush=True)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set in environment — cannot publish.", file=sys.stderr)
        sys.exit(2)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=args.private, exist_ok=True)

    card = build_model_card(source_model=source_model, meta=meta, gate=gate)
    card = card.replace("{this_repo_id}", repo_id)
    with open(os.path.join(args.local_dir, "README.md"), "w") as f:
        f.write(card)

    api.upload_folder(repo_id=repo_id, folder_path=args.local_dir,
                       ignore_patterns=["_fastserve_*"])
    print(f"PUBLISHED: https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
