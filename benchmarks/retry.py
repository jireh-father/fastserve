"""One-off retry pass for models that failed in the main sweep for reasons
now fixed (detect() false-positive repos, missing trust_remote_code,
transient HF rate-limit) or need a targeted workaround demonstrated
(GLM-4.7-Flash's EAGLE3-incompatible architecture -> --no-spec).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver import run_one, ensure_header  # noqa: E402

ensure_header()

PLAIN_RETRIES = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.6-27B",
    "openai/gpt-oss-20b",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

for m in PLAIN_RETRIES:
    run_one(m)

# gemma-4-12B-it: only the long-thinker MMLU/GSM8K budget changed; baseline
# speed number from the first pass is still valid, don't re-measure it.
run_one("google/gemma-4-12B-it", reuse_baseline=True)

# GLM-4.7-Flash: vLLM's implementation doesn't support the EAGLE3 interface
# for this architecture (confirmed in logs/zai-org__GLM-4.7-Flash.log) —
# retry with speculative decoding fully disabled to demonstrate the
# workaround rather than silently leaving it as an unexplained failure.
run_one("zai-org/GLM-4.7-Flash", extra_args=["--no-spec"])
