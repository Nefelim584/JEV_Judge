"""Speed of jev-lite by context length, precision and attention backend (todo Phase 9).

The grid: every context length × every variant, on one GPU per call (``scripts/modal_latency.py`` runs
several GPUs in parallel). Default variants (FlashAttention has no fp32 kernels, so "fp32 + FA" does not
exist; FA3 needs Hopper, sm90, and is reported as n/a elsewhere):

    fp32-sdpa   fp32 weights, PyTorch SDPA (no FlashAttention)
    bf16-sdpa   bf16 weights, PyTorch SDPA
    bf16-fa2    bf16 weights, FlashAttention 2 (Hub kernel ``kernels-community/flash-attn2``)
    bf16-fa3    bf16 weights, FlashAttention 3 (Hub kernel ``kernels-community/flash-attn3``)

Per cell (length × variant):

- **response time** (latency) of one request, ``n_runs`` calls after ``n_warmup`` untimed ones:
  ``e2e_ms`` = ``Predictor.predict`` (tokenisation, forward, calibration, typed answer) and
  ``forward_ms`` = the model's forward pass alone (CUDA-synchronised). Full statistics
  (p50 / p90 / p95 / p99, mean, std, min, max) and the raw samples are kept.
- **throughput**: ``throughput_batch`` requests of that length in one call (each tokenised on its own),
  ``n_throughput`` timed calls: requests/s and tokens/s. Halved on out-of-memory; the batch used is recorded.
- **max_abs_dp**: the largest |P − P(fp32-sdpa)| on the same request, so a speed-up that moves the
  answers shows.

A request is one Bool claim on a filler document sized to exactly fill the length (the cost depends on
the number of tokens, not on the words).
"""

from __future__ import annotations

import gc
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .calibration import load_calibration
from .collate import build_batch
from .encoding import text_encoder_from_config
from .inference import Predictor
from .schema import BoolQuestion, Request

DEFAULT_LENGTHS = (512, 2048, 4096, 8192)
MAX_LEN = 8192
TARGET_MS = 100.0

_FILLER = (
    "The order entry gateway accepts a NewOrderSingle message only after a successful Logon on the same session.",
    "Each request carries a client order id that must stay unique for the trading day.",
    "If the price field is missing on a limit order, the gateway rejects it with reason code 11.",
    "The REST endpoint returns the order status together with the cumulative filled quantity.",
    "Rate limits apply per API key: at most twenty requests per second, with bursts of up to forty.",
    "A cancel request refers to the original order by its client order id and the instrument symbol.",
    "Heartbeats are sent every thirty seconds; two missed heartbeats close the session.",
    "Pagination uses an opaque cursor; the client must not build cursors itself.",
)
_CLAIMS = (
    "The gateway rejects a limit order without a price.",
    "Rate limits are applied per API key.",
    "Two missed heartbeats close the session.",
    "Order status includes the cumulative filled quantity.",
)


@dataclass(frozen=True)
class Variant:
    name: str
    dtype: str = "fp32"  # weights: fp32 | bf16
    attn: str = "sdpa"  # sdpa | eager | flex_attention | flash_attention_2 | kernels-community/<repo>
    amp: str | None = None  # autocast dtype over fp32 weights: bf16 | None
    compile: bool = False  # torch.compile of the encoder
    min_capability: tuple[int, int] | None = None  # e.g. (9, 0): Hopper only
    max_capability: tuple[int, int] | None = None


FA2 = "kernels-community/flash-attn2"
FA3 = "kernels-community/flash-attn3"
DEFAULT_VARIANTS = (
    Variant("fp32-sdpa"),
    Variant("bf16-sdpa", dtype="bf16"),
    Variant("bf16-fa2", dtype="bf16", attn=FA2, min_capability=(8, 0)),
    Variant("bf16-fa3", dtype="bf16", attn=FA3, min_capability=(9, 0), max_capability=(9, 0)),
)
# Not in the default grid; run by name (``--variants``).
EXTRA_VARIANTS = (
    Variant("amp-bf16-sdpa", amp="bf16"),
    Variant("bf16-flex", dtype="bf16", attn="flex_attention"),
    Variant("bf16-sdpa-compile", dtype="bf16", compile=True),
    Variant("bf16-fa2-compile", dtype="bf16", attn=FA2, min_capability=(8, 0), compile=True),
)
BASELINE = "fp32-sdpa"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, None: None}


def variants_by_name(names: Sequence[str] | None) -> list[Variant]:
    known = {v.name: v for v in DEFAULT_VARIANTS + EXTRA_VARIANTS}
    if not names:
        return list(DEFAULT_VARIANTS)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown variants {unknown}, expected some of {list(known)}")
    return [known[n] for n in names]


def unsupported_reason(variant: Variant, capability: tuple[int, int] | None) -> str | None:
    """Why the variant cannot run on a GPU of this compute capability (``None``: it can)."""
    if capability is None:
        return "needs CUDA" if variant.min_capability or variant.attn.startswith("kernels-community/") else None
    if variant.min_capability and capability < variant.min_capability:
        return f"needs sm{variant.min_capability[0]}{variant.min_capability[1]}+, GPU is sm{capability[0]}{capability[1]}"
    if variant.max_capability and capability > variant.max_capability:
        return f"needs sm{variant.max_capability[0]}{variant.max_capability[1]}, GPU is sm{capability[0]}{capability[1]}"
    return None


# ---------------------------------------------------------------------- requests of a given length


def _filler(n_sentences: int) -> str:
    return " ".join(f"{i + 1}. {_FILLER[i % len(_FILLER)]}" for i in range(n_sentences))


def _request(n_sentences: int, n_claims: int) -> Request:
    claims = [BoolQuestion(id=f"c{i}", prompt=_CLAIMS[i % len(_CLAIMS)]) for i in range(n_claims)]
    return Request(state={"document": _filler(n_sentences)}, questions=claims)


def _length(request: Request, encoder) -> tuple[int, bool]:
    batch = build_batch([(request.state, q) for q in request.questions], encoder)
    return int(batch.input_ids.shape[1]), bool(batch.truncated.any())


def fit_request(encoder, n_tokens: int, n_claims: int = 1) -> tuple[Request, int]:
    """The longest filler request whose packed sequences fit in ``n_tokens`` untruncated, and its length.

    ``encoder.max_len`` must be at least ``n_tokens``."""
    if encoder.max_len < n_tokens:
        raise ValueError(f"encoder max_len {encoder.max_len} < {n_tokens}")
    lo, hi = 1, 1
    while True:  # grow until too long
        length, truncated = _length(_request(hi, n_claims), encoder)
        if truncated or length > n_tokens:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:  # lo fits, hi does not
        mid = (lo + hi) // 2
        length, truncated = _length(_request(mid, n_claims), encoder)
        lo, hi = (mid, hi) if not truncated and length <= n_tokens else (lo, mid)
    request = _request(lo, n_claims)
    length, truncated = _length(request, encoder)
    if truncated or length > n_tokens:
        raise ValueError(f"no untruncated request fits in {n_tokens} tokens")
    return request, length


# ---------------------------------------------------------------------- timing


def summarize(ms: Sequence[float]) -> dict:
    v = np.asarray(ms, dtype=float)
    out = {f"p{q}": float(np.percentile(v, q)) for q in (50, 90, 95, 99)}
    return out | {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_variant(final_dir: str | Path, variant: Variant, device: str | torch.device, max_len: int = MAX_LEN) -> Predictor:
    """The release predictor (calibration, model layer) in the variant's setting."""
    from .checkpoint import load_final

    model, tokenizer, cfg = load_final(final_dir, device=device, attn_implementation=variant.attn)
    if variant.dtype != "fp32":
        model = model.to(_DTYPES[variant.dtype])
    if variant.compile:
        model.encoder = torch.compile(model.encoder)
    return Predictor(model, text_encoder_from_config(cfg, tokenizer, max_len), name=variant.name,
                     amp_dtype=_DTYPES[variant.amp], calibration=load_calibration(final_dir))


def time_latency(predictor: Predictor, request: Request, n_warmup: int, n_runs: int) -> dict:
    """Response time of one request: end to end and forward only, with the raw samples."""
    device = predictor.device
    for _ in range(n_warmup):
        predictor.predict(request)
    first = predictor.predict(request)
    probs = [first.answers[q.id].probability for q in request.questions]
    e2e = [predictor.predict(request).meta.latency_ms for _ in range(n_runs)]

    batch = build_batch([(request.state, q) for q in request.questions], predictor.encoder).to(device)
    forward = []
    with torch.no_grad(), torch.autocast(device.type, dtype=predictor.amp_dtype, enabled=predictor.amp_dtype is not None):
        for _ in range(n_runs):
            _sync(device)
            t = time.perf_counter()
            predictor.model.grouped_logits(batch)
            _sync(device)
            forward.append((time.perf_counter() - t) * 1000)
    return {"e2e_ms": summarize(e2e), "forward_ms": summarize(forward), "probs": probs,
            "samples": {"e2e_ms": e2e, "forward_ms": forward}}


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def time_throughput(predictor: Predictor, request: Request, n_tokens: int, batch: int, n_warmup: int, n_runs: int) -> dict:
    """``batch`` requests of one length per call, each tokenised on its own; halves the batch on OOM."""
    device = predictor.device
    question = request.questions[0]
    while batch >= 1:
        items = [(dict(request.state), question) for _ in range(batch)]  # distinct states: no shared tokenisation
        try:
            for _ in range(n_warmup):
                predictor._probs(items)
            times = []
            for _ in range(n_runs):
                _sync(device)
                t = time.perf_counter()
                predictor._probs(items)
                _sync(device)
                times.append(time.perf_counter() - t)
            s = summarize([x * 1000 for x in times])
            per_s = batch / (s["p50"] / 1000)
            return {"batch": batch, "call_ms": s, "requests_per_s": per_s, "tokens_per_s": per_s * n_tokens,
                    "samples_ms": [x * 1000 for x in times]}
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e) or batch == 1:
                raise
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch //= 2
    raise RuntimeError("unreachable")


def environment(device: torch.device) -> dict:
    import transformers

    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env |= {"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 2**30, 1),
                "capability": [props.major, props.minor], "cuda": torch.version.cuda}
    return env


def run_benchmark(final_dir: str | Path, variants: Sequence[Variant], lengths: Sequence[int], device: str | torch.device,
                  n_warmup: int = 10, n_runs: int = 100, throughput_batch: int = 8, n_throughput: int = 20,
                  log=print) -> dict:
    """Every variant on every length. A variant that cannot run on this GPU is marked ``skipped``; one
    that fails is recorded with its error; either way the others go on."""
    device = torch.device(device)
    env = environment(device)
    capability = tuple(env["capability"]) if "capability" in env else None
    requests: dict[int, tuple[Request, int]] = {}
    out = {"env": env, "settings": {"n_warmup": n_warmup, "n_runs": n_runs, "throughput_batch": throughput_batch,
                                    "n_throughput": n_throughput, "max_len": MAX_LEN}, "lengths": {}, "variants": {}}

    for variant in variants:
        row = {"variant": asdict(variant), "lengths": {}}
        out["variants"][variant.name] = row
        if reason := unsupported_reason(variant, capability):
            row["skipped"] = reason
            log(f"{variant.name:>14}: n/a ({reason})")
            continue
        predictor = None
        try:
            t = time.perf_counter()
            predictor = load_variant(final_dir, variant, device)
            row["load_s"] = time.perf_counter() - t
            row["attn_in_use"] = getattr(predictor.model.encoder.config, "_attn_implementation", None)
            if not requests:  # requests depend only on the tokenizer: fit once
                requests = {n: fit_request(predictor.encoder, n) for n in lengths}
                out["lengths"] = {str(n): {"tokens": k} for n, (_, k) in requests.items()}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for n in lengths:
                t = time.perf_counter()
                request, tokens = requests[n]
                cell = time_latency(predictor, request, n_warmup, n_runs)
                cell["throughput"] = time_throughput(predictor, request, tokens, throughput_batch, min(n_warmup, 3), n_throughput)
                row["lengths"][str(n)] = cell
                e, th = cell["e2e_ms"], cell["throughput"]
                log(f"{variant.name:>14} {n:>5}: e2e p50 {e['p50']:7.1f} ms p95 {e['p95']:7.1f} (±{e['std']:.1f}); "
                    f"forward p50 {cell['forward_ms']['p50']:7.1f}; {th['requests_per_s']:7.1f} req/s at batch {th['batch']} "
                    f"({time.perf_counter() - t:.0f} s)")
            if device.type == "cuda":
                row["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        except Exception as e:  # noqa: BLE001 — record and go on with the next variant
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc(limit=8)
            log(f"{variant.name:>14}: FAILED {row['error']}")
        finally:
            del predictor
            gc.collect()
            if variant.compile:
                torch._dynamo.reset()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    base = out["variants"].get(BASELINE, {}).get("lengths", {})
    for row in out["variants"].values():
        for n, cell in row["lengths"].items():
            if n in base:
                cell["max_abs_dp"] = float(max(abs(a - b) for a, b in zip(cell["probs"], base[n]["probs"], strict=True)))
    return out


# ---------------------------------------------------------------------- reports


def _gpu_name(res: dict) -> str:
    return res.get("env", {}).get("gpu") or res.get("modal_gpu") or res["env"]["device"]


def rows_long(results: Sequence[dict]) -> list[dict]:
    """One flat row per GPU × variant × length (CSV): latency stats, throughput, Δp."""
    rows = []
    for res in results:
        for name, row in res["variants"].items():
            base = {"gpu": _gpu_name(res), "variant": name, "dtype": row["variant"]["dtype"], "attn": row["variant"]["attn"]}
            if "skipped" in row or "error" in row:
                rows.append(base | {"status": row.get("skipped") or row.get("error")})
                continue
            for n, cell in row["lengths"].items():
                r = base | {"status": "ok", "length": int(n), "tokens": res["lengths"][n]["tokens"]}
                for metric in ("e2e_ms", "forward_ms"):
                    r |= {f"{metric}_{k}": v for k, v in cell[metric].items()}
                th = cell["throughput"]
                r |= {"throughput_batch": th["batch"], "requests_per_s": th["requests_per_s"], "tokens_per_s": th["tokens_per_s"],
                      "max_abs_dp": cell.get("max_abs_dp"), "peak_memory_gb": row.get("peak_memory_gb")}
                rows.append(r)
    return rows


def _table(res: dict, title: str, fmt) -> list[str]:
    names = list(res["variants"])
    lines = [f"**{title}**", "", "| length | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for n in res["lengths"]:
        cells = []
        for name in names:
            row = res["variants"][name]
            cells.append("n/a" if "skipped" in row else "error" if "error" in row else fmt(row["lengths"][n], row))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return lines + [""]


def report_markdown(results: Sequence[dict], target_ms: float = TARGET_MS) -> str:
    """Per GPU: response time, forward, throughput, Δp by length × variant; then 8k across GPUs."""
    lines = []
    for res in results:
        env, s = res["env"], res["settings"]
        lines += [f"## {_gpu_name(res)}", "",
                  f"torch {env['torch']}, transformers {env['transformers']}, CUDA {env.get('cuda')}. "
                  f"Response time: {s['n_runs']} calls after {s['n_warmup']} warm-up; throughput: {s['n_throughput']} calls "
                  f"of up to {s['throughput_batch']} requests. ✓: p95 ≤ {target_ms:g} ms.", ""]
        lines += _table(res, "Response time, end to end, ms: p50 / p95 (std)",
                        lambda c, r: f"{c['e2e_ms']['p50']:.1f} / {c['e2e_ms']['p95']:.1f} ({c['e2e_ms']['std']:.1f})"
                                     + (" ✓" if c["e2e_ms"]["p95"] <= target_ms else ""))
        lines += _table(res, "Forward pass only, ms: p50 / p95",
                        lambda c, r: f"{c['forward_ms']['p50']:.1f} / {c['forward_ms']['p95']:.1f}")
        lines += _table(res, "Throughput: requests/s (tokens/s, batch)",
                        lambda c, r: f"{c['throughput']['requests_per_s']:.1f} ({c['throughput']['tokens_per_s'] / 1000:.0f}k, "
                                     f"{c['throughput']['batch']})")
        lines += _table(res, "Max |ΔP| vs fp32-sdpa",
                        lambda c, r: f"{c.get('max_abs_dp', float('nan')):.4f}")
        for name, row in res["variants"].items():
            if "skipped" in row or "error" in row:
                lines.append(f"- {name}: {row.get('skipped') or row.get('error')}")
        lines.append("")

    longest = max((max(map(int, r["lengths"]), default=0) for r in results), default=0)
    if results and longest:
        names = list(dict.fromkeys(n for r in results for n in r["variants"]))
        lines += [f"## All GPUs at {longest} tokens: response time p50 / p95, ms", "",
                  "| GPU | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for res in results:
            cells = []
            for name in names:
                row = res["variants"].get(name, {})
                cell = row.get("lengths", {}).get(str(longest))
                cells.append("n/a" if "skipped" in row else "—" if cell is None else
                             f"{cell['e2e_ms']['p50']:.1f} / {cell['e2e_ms']['p95']:.1f}"
                             + (" ✓" if cell["e2e_ms"]["p95"] <= target_ms else ""))
            lines.append(f"| {_gpu_name(res)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
