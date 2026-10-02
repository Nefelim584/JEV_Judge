"""Zero-shot baselines (TODO Phase 2): an NLI cross-encoder and Laya. Same interface as our own model."""

BASELINES = ("nli", "laya")


def load_baseline(name: str, model: str | None = None, device: str | None = None, max_len: int = 512):
    """``nli``: ``model`` is a Hugging Face NLI checkpoint. ``laya``: base | multilingual | typed-decisions."""
    if name == "nli":
        from .nli import DEFAULT_MODEL, NLIZeroShot

        return NLIZeroShot(model or DEFAULT_MODEL, device=device, max_len=max_len)
    if name == "laya":
        from .laya import LayaZeroShot

        return LayaZeroShot(model or "base", device=device)
    raise ValueError(f"unknown baseline {name!r}, expected one of {BASELINES}")
