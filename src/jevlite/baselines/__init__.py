"""Zero-shot baselines (TODO Phase 2): an NLI cross-encoder and Laya. Same interface as our own model."""

BASELINES = ("nli", "laya")


def load_baseline(
    name: str, model: str | None = None, device: str | None = None, max_len: int = 512,
    batch_size: int | None = None, fp16: bool = False,
):
    """``nli``: ``model`` is a Hugging Face NLI checkpoint. ``laya``: base | multilingual | typed-decisions.

    ``batch_size`` (premise–hypothesis pairs per forward pass) and ``fp16`` apply to NLI only; Laya
    already runs one batch per state and uses mixed precision on CUDA by itself.
    """
    if name == "nli":
        from .nli import DEFAULT_MODEL, NLIZeroShot

        kw = {"batch_size": batch_size} if batch_size else {}
        return NLIZeroShot(model or DEFAULT_MODEL, device=device, max_len=max_len, fp16=fp16, **kw)
    if batch_size or fp16:
        raise ValueError(f"--batch-size and --fp16 apply to the nli baseline only, not {name!r}")
    if name == "laya":
        from .laya import LayaZeroShot

        return LayaZeroShot(model or "base", device=device)
    raise ValueError(f"unknown baseline {name!r}, expected one of {BASELINES}")
