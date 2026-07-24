"""HF Hub auto-detection: find pre-quantized (AWQ/GPTQ) checkpoints and
EAGLE-3 speculative-decoding draft heads for a given base model.

Naming conventions and publisher orgs below were validated empirically
against real HF Hub listings during the inference-opt research campaign
(see ../../../inference-opt/RESULTS.md and CATALOG.md for the full writeup).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

_api = HfApi()

# Publishers known to release EAGLE-3 draft heads for popular base models.
_EAGLE_ORGS = ["AngelSlim", "yuhuili", "SpecForge", "NousResearch"]
_EAGLE_NAME_PATTERNS = ["{base}_eagle3", "{base}-eagle3", "EAGLE3-{base}", "{base}_eagle"]

_QUANT_SUFFIXES = ["AWQ", "GPTQ", "W8A8-INT8"]


def _method_of(repo_id: str) -> str:
    """Coarse quant-method label from the repo name (vLLM reads the real scheme
    from the checkpoint's compressed-tensors config; this is just for display)."""
    low = repo_id.lower()
    if "awq" in low:
        return "awq"
    if "w8a8" in low or "int8" in low:
        return "w8a8"
    return "gptq"

# Namespace(s) checked before any generic org-guess or broad-search candidate.
# The idea: publish your own accuracy-gated quants (see fastserve/publish/) to
# a namespace you control, list it here, and fastserve prefers it over an
# arbitrary community requant — benchmarks/RESULTS.md has two real cases
# (gemma-2-2b/9b-it) where trusting the latter silently served a broken model.
# Override with FASTSERVE_TRUSTED_NAMESPACES=ns1,ns2 (defaults to the
# maintainer's published set).
_TRUSTED_NAMESPACES = [
    ns.strip()
    for ns in os.environ.get("FASTSERVE_TRUSTED_NAMESPACES", "glenic").split(",")
    if ns.strip()
]


@dataclass
class DetectionResult:
    base_model: str
    quantized_model: str | None = None
    quant_method: str | None = None  # "awq" | "gptq" | None
    eagle_model: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_quant(self) -> bool:
        return self.quantized_model is not None

    @property
    def has_eagle(self) -> bool:
        return self.eagle_model is not None


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def _repo_exists(repo_id: str) -> bool:
    return _repo_downloadable(repo_id)


def _repo_downloadable(repo_id: str) -> bool:
    """`model_info()` succeeding isn't enough — a repo can exist but still be
    unusable: gated (needs a token we don't have) or an incomplete/empty
    upload with no config.json (vLLM then fails with "Invalid repository
    ID"). Both were observed in practice picking candidates for very recently
    released models, where the community hasn't fully populated repos yet.
    """
    try:
        info = _api.model_info(repo_id)
    except (HfHubHTTPError, Exception):
        return False
    if info.gated:
        return False
    return any(s.rfilename == "config.json" for s in (info.siblings or []))


def find_quantized(model_id: str) -> tuple[str | None, str | None]:
    """Search for a pre-quantized checkpoint of model_id.

    Returns (repo_id, method) or (None, None) if nothing was found.
    """
    org = model_id.split("/", 1)[0] if "/" in model_id else None
    name = _short_name(model_id)

    trusted = [f"{ns}/{name}-{suf}" for ns in _TRUSTED_NAMESPACES for suf in _QUANT_SUFFIXES]
    for c in trusted:
        if _repo_exists(c):
            return c, _method_of(c)

    candidates: list[str] = []
    for suf in _QUANT_SUFFIXES:
        if org:
            candidates.append(f"{org}/{name}-{suf}")
        candidates.append(f"{name}-{suf}")
    for c in candidates:
        if _repo_exists(c):
            return c, _method_of(c)

    # Broader hub search fallback (catches community requants under other orgs).
    try:
        for suf in _QUANT_SUFFIXES:
            hits = _api.list_models(search=f"{name} {suf}", limit=15)
            for h in hits:
                low = h.id.lower()
                if name.lower() in low and suf.lower() in low and _repo_downloadable(h.id):
                    return h.id, suf.lower()
    except Exception:
        pass
    return None, None


def _eagle_config_ok(repo_id: str) -> bool:
    """A found EAGLE head is only useful if vLLM can actually load it. Some
    community heads ship a self-inconsistent config vLLM rejects at engine start
    (observed: `Dogacel/specdrift-qwen3.6-27b-eagle3`, hidden_size 5120 not
    divisible by 24 attention heads). Cheaply pre-validate the head's own config
    so detect() falls back to n-gram instead of handing serve a head that crashes
    the whole engine. Unknown/uncheckable → don't block (assume usable)."""
    try:
        import json as _json

        from huggingface_hub import hf_hub_download
        cfg = _json.load(open(hf_hub_download(repo_id, "config.json")))
        for key in ("model", "draft_model"):  # some heads nest the real config
            if isinstance(cfg.get(key), dict):
                cfg = {**cfg, **cfg[key]}
        h, a = cfg.get("hidden_size"), cfg.get("num_attention_heads")
        if h and a and h % a != 0:
            return False
    except Exception:
        pass
    return True


def find_eagle3(model_id: str) -> str | None:
    """Search for a published EAGLE-3 draft head compatible with model_id."""
    name = _short_name(model_id)

    candidates = [
        f"{org}/{pat.format(base=name)}"
        for org in _EAGLE_ORGS
        for pat in _EAGLE_NAME_PATTERNS
    ]
    for c in candidates:
        if _repo_exists(c) and _eagle_config_ok(c):
            return c

    try:
        hits = _api.list_models(search=f"{name} eagle3", limit=15)
        for h in hits:
            low = h.id.lower()
            if name.lower() in low and "eagle" in low and _repo_downloadable(h.id) \
                    and _eagle_config_ok(h.id):
                return h.id
    except Exception:
        pass
    return None


def detect(model_id: str, *, skip_quant: bool = False, skip_eagle: bool = False) -> DetectionResult:
    """Run the full auto-detection pipeline for a model.

    Never raises: any lookup failure just means that optimization is skipped
    and falls back to the safe default (original precision / n-gram spec).
    """
    res = DetectionResult(base_model=model_id)

    if not skip_quant:
        q_repo, q_method = find_quantized(model_id)
        if q_repo:
            res.quantized_model, res.quant_method = q_repo, q_method
            res.notes.append(f"found pre-quantized checkpoint: {q_repo} ({q_method.upper()})")
        else:
            res.notes.append("no pre-quantized checkpoint found -> serving at original precision")

    if not skip_eagle:
        eagle = find_eagle3(model_id)
        if eagle:
            res.eagle_model = eagle
            res.notes.append(f"found EAGLE-3 draft head: {eagle}")
        else:
            res.notes.append(
                "no EAGLE-3 draft head found -> falling back to n-gram speculative "
                "decoding (zero-cost, no extra download, still lossless under greedy)"
            )

    return res
