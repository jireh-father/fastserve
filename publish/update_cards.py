"""Regenerate + re-upload the README (model card) for every published quant,
now including the GitHub link, a fastserve serving manual, and the 3-way
original/vLLM/fastserve comparison table (from benchmarks/results/cmp_*.json).

Pulls the quant meta + gate result from each published repo's own
`_fastserve_gate_result.json`... except that file is gitignore-excluded from
upload, so instead we reconstruct meta/gate from the gate JSON left in the
local run, falling back to reading the published config. To stay simple and
robust, this reads the per-model comparison JSON and the recorded gate
numbers from a small hand-maintained map when the local gate json is gone.
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from model_card import build_model_card  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
NS = os.environ.get("FASTSERVE_HF_NAMESPACE", "seoilgun")
RESULTS = os.path.join(ROOT, "benchmarks", "results")

# base model id -> (repo short name, bf16 baseline acc, our AWQ acc, calib dataset)
# baseline/quant accuracy are the n=30 gate numbers recorded during publishing.
MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B-Instruct-AWQ", 0.3333, 0.3333, "ultrachat-200k"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B-Instruct-AWQ", 0.5667, 0.5667, "ultrachat-200k"),
    "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct-AWQ", 0.8667, 0.8667, "ultrachat-200k"),
    "Qwen/Qwen2.5-14B-Instruct": ("Qwen2.5-14B-Instruct-AWQ", 0.9667, 0.9333, "ultrachat-200k"),
    "Qwen/Qwen2.5-32B-Instruct": ("Qwen2.5-32B-Instruct-AWQ", 0.9333, 0.9333, "ultrachat-200k"),
    "Qwen/Qwen3-8B": ("Qwen3-8B-AWQ", 0.8333, 0.8333, "ultrachat-200k"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("Mistral-7B-Instruct-v0.3-AWQ", 0.4000, 0.3333, "ultrachat-200k"),
    "google/gemma-2-2b-it": ("gemma-2-2b-it-AWQ", 0.5333, 0.4667, "wikitext"),
    "google/gemma-2-9b-it": ("gemma-2-9b-it-AWQ", 0.7000, 0.6333, "wikitext"),
    "01-ai/Yi-1.5-9B-Chat": ("Yi-1.5-9B-Chat-AWQ", 0.4333, 0.6000, "ultrachat-200k"),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ("DeepSeek-R1-Distill-Qwen-7B-AWQ", 0.7667, 0.8667, "ultrachat-200k"),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ("DeepSeek-R1-0528-Qwen3-8B-AWQ", 0.5333, 0.7333, "ultrachat-200k"),
}


def main() -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    dry = "--dry-run" in sys.argv

    for base, (repo_short, base_acc, quant_acc, calib) in MODELS.items():
        repo_id = f"{NS}/{repo_short}"
        meta = {"model": base, "baseline": {"n": 30, "acc": base_acc},
                "quant_config": {"scheme": "W4A16_ASYM", "group_size": 128,
                                 "calib_dataset": calib, "calib_samples": 256}}
        gate = {"quant_acc": quant_acc, "max_drop": 0.10}

        cmp_path = os.path.join(RESULTS, f"cmp_{base.split('/')[-1]}.json")
        comparison = json.load(open(cmp_path)) if os.path.exists(cmp_path) else None
        if comparison is None:
            print(f"  (no comparison data yet for {base} — table omitted)", flush=True)

        card = build_model_card(base, meta, gate, comparison).replace("{this_repo_id}", repo_id)
        if dry:
            print(f"--- {repo_id} ({'with' if comparison else 'no'} table) ---")
            continue
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo_id, commit_message="Add fastserve link, serving guide, comparison table")
        print(f"  updated {repo_id}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
