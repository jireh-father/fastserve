"""Candidate model list for the fastserve cross-model benchmark.

Chosen to span major open families and a 0.5B-108B size range, restricted to
models where a public (non-gated) AWQ/GPTQ requant exists on the Hub so the
whole sweep runs without an HF token (verified via `detect()` dry-run before
committing this list).

MODELS_2024_2025 = first pass (2026-07-05): still-common families, but by
"now" (2026-07) already 1-2 generations behind each vendor's latest release.
MODELS_CURRENT = added on request ("test genuinely current models too") after
checking the HF Hub directly for what each lab has shipped most recently.
"""

MODELS_2024_2025 = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

MODELS_CURRENT = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-12B-it",
    # mistralai/Ministral-3-8B-Instruct-2512 dropped: Mistral's entire current
    # generation (Ministral-3/Devstral-2/Magistral, all Dec 2025) unified on
    # Mistral3ForConditionalGeneration — a vision-language architecture, not
    # a fit for this text-only harness (also hung under vLLM+ngram spec).
    "openai/gpt-oss-20b",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "zai-org/GLM-4.7-Flash",
    "allenai/tmax-sft-8b",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

MODELS = MODELS_2024_2025 + MODELS_CURRENT

# Gated base repos where the un-quantized bf16 weights need an HF token —
# point the baseline (naive HF-eager, unquantized) measurement at a verified
# public mirror instead so the whole sweep still needs no token.
BASELINE_MIRROR = {
    "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it": "unsloth/gemma-2-2b-it",
    "google/gemma-2-9b-it": "unsloth/gemma-2-9b-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "unsloth/Llama-4-Scout-17B-16E-Instruct",
}

# Models whose bf16 (unquantized) weights don't fit an 80GB A100 at all —
# no point downloading tens/hundreds of GB just to OOM. AWQ is precisely
# what makes serving these feasible on this hardware in the first place.
BASELINE_SKIP = {
    "Qwen/Qwen2.5-72B-Instruct": "bf16 needs ~144GB > single 80GB GPU",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "bf16 needs ~98GB (49B total params) > single 80GB GPU",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "bf16 needs ~217GB (108B total params) > single 80GB GPU",
}
