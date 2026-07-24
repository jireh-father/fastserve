"""Assemble and launch a vLLM engine (offline benchmark or OpenAI-compatible
server) using the best detected configuration for a given model.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys

from .detect import DetectionResult


def build_speculative_config(det: DetectionResult, spec_tokens: int = 3) -> dict:
    """Pick the best available speculative-decoding config.

    EAGLE-3 if a matching draft head was found; otherwise n-gram (prompt
    lookup), which needs no extra download and still helps on any workload
    with repeated substrings (code, structured output, long context).
    """
    if det.eagle_model:
        return {"method": "eagle3", "model": det.eagle_model, "num_speculative_tokens": spec_tokens}
    return {
        "method": "ngram",
        "num_speculative_tokens": spec_tokens,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    }


def resolve_model(det: DetectionResult, use_quant: bool) -> str:
    if use_quant and det.quantized_model:
        return det.quantized_model
    return det.base_model


def serve_command(
    det: DetectionResult,
    *,
    port: int = 8000,
    use_quant: bool = True,
    use_spec: bool = True,
    tp: int = 1,
    trust_remote_code: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    model = resolve_model(det, use_quant)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
    ]
    if use_spec:
        cmd += ["--speculative-config", json.dumps(build_speculative_config(det))]
    if trust_remote_code:
        # Some checkpoints (e.g. Kimi-Linear's tokenizer) ship custom code
        # that transformers/vLLM refuse to run without this. On by default
        # since you already chose to serve this specific model id; opt out
        # with --no-trust-remote-code if that's not an acceptable trade-off.
        cmd += ["--trust-remote-code"]
    if extra_args:
        cmd += extra_args
    return cmd


def launch_server(det: DetectionResult, **kwargs) -> None:
    cmd = serve_command(det, **kwargs)
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    subprocess.run(cmd, check=False)
