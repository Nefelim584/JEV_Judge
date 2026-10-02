"""Sampling mix: which converted records a training stage sees (TODO Phase 3, ``DATASETS.md`` section 0).

A mix config (``configs/mixes/*.yaml``) names the sources, a cap per source and the share of every
primitive in the training data. Building a mix has two steps:

1. **Pools.** For every source, the training records that pass its filters, capped to ``cap`` by a
   stable hash of the record id, so the same cap always keeps the same records.
   ``fit_only: true`` keeps only records whose state fits ``max_len`` untruncated (ratings, where
   truncation can hide the part the label is about), in the eval splits too.
2. **Epochs.** ``total`` training records per epoch, ``primitive_weights[p] · total`` of primitive
   ``p``, split between the pools of ``p`` in proportion to ``pool size × weight``. A pool smaller
   than its quota is repeated (at most ``max_repeat`` times, else the build fails); a larger one is
   subsampled, with a fresh draw every epoch.

Eval splits (``calib``, ``test_in``, ``test_ood``) are not weighted: every source keeps up to
``eval_cap`` records per split, the same ones every time.

Sources are read line by line and only the kept lines stay in memory, so a 3M-record mix builds on
a 16 GB laptop.
"""

from __future__ import annotations

import heapq
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import yaml

from ..schema import PRIM_TYPES
from .public.base import stable_hash
from .unified import Record

EVAL_SPLITS = ("calib", "test_in", "test_ood")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    cap: int | None = None
    weight: float = 1.0
    fit_only: bool = False
    eval_cap: int | None = None  # overrides the mix-level eval_cap


@dataclass(frozen=True)
class MixConfig:
    name: str
    dir: Path
    total: int
    primitive_weights: dict[str, float]
    sources: tuple[SourceSpec, ...]
    max_repeat: float = 2.0
    eval_cap: int | None = 2000
    max_len: int = 512
    tokenizer: str = "answerdotai/ModernBERT-large"
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict, name: str = "mix") -> MixConfig:
        weights = {p: float(w) for p, w in d["primitive_weights"].items()}
        if set(weights) - set(PRIM_TYPES) or any(w < 0 for w in weights.values()):
            raise ValueError(f"primitive_weights must map {PRIM_TYPES} to non-negative weights, got {weights}")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"primitive_weights must sum to 1, got {sum(weights.values())}")
        sources = tuple(SourceSpec(name=n, **(spec or {})) for n, spec in d["sources"].items())
        return cls(
            name=d.get("name", name), dir=Path(d["dir"]), total=int(d["total"]), primitive_weights=weights,
            sources=sources, max_repeat=float(d.get("max_repeat", 2.0)), eval_cap=d.get("eval_cap", 2000),
            max_len=int(d.get("max_len", 512)), tokenizer=d.get("tokenizer", cls.tokenizer), seed=int(d.get("seed", 42)),
        )


def load_mix_config(path: str | Path) -> MixConfig:
    path = Path(path)
    with path.open() as f:
        return MixConfig.from_dict(yaml.safe_load(f), name=path.stem)


# ---------------------------------------------------------------------------------------------------
# Step 1: pools


@dataclass
class Pools:
    """Kept JSONL lines per (source, primitive) for training, and per (split, source) for eval."""

    train: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    eval: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Training records seen before the cap / dropped by the fit filter, per source.
    available: dict[str, int] = field(default_factory=dict)
    dropped_unfit: dict[str, int] = field(default_factory=dict)


class _Capped:
    """The ``cap`` items with the smallest hash keys, in O(cap) memory (``cap=None`` keeps all)."""

    def __init__(self, cap: int | None):
        self.cap = cap
        self.heap: list[tuple[int, str, str]] = []  # (-key, type, line): a max-heap on key
        self.all: list[tuple[str, str]] = []

    def add(self, key: int, prim: str, line: str) -> None:
        if self.cap is None:
            self.all.append((prim, line))
        elif len(self.heap) < self.cap:
            heapq.heappush(self.heap, (-key, prim, line))
        elif -self.heap[0][0] > key:
            heapq.heapreplace(self.heap, (-key, prim, line))

    def items(self) -> list[tuple[str, str]]:
        if self.cap is None:
            return self.all
        return [(prim, line) for _, prim, line in sorted(self.heap, key=lambda x: -x[0])]


def fits_checker(tokenizer_name: str, max_len: int) -> Callable[[dict], bool]:
    """``row → True`` if the record's packed sequence needs no truncation at ``max_len``."""
    from transformers import AutoTokenizer

    from ..encoding import PackedEncoder

    enc = PackedEncoder(AutoTokenizer.from_pretrained(tokenizer_name), max_len)

    def fits(row: dict) -> bool:
        r = Record.model_validate(row)
        (seq,) = enc.encode_item(r.to_question(), enc.tokenize_state(r.state))
        return not seq.truncated

    return fits


def build_pools(cfg: MixConfig, fits: Callable[[dict], bool] | None = None) -> Pools:
    """Read every source once and keep its capped training pool and its eval records."""
    pools = Pools()
    if fits is None and any(s.fit_only for s in cfg.sources):
        fits = fits_checker(cfg.tokenizer, cfg.max_len)
    for spec in cfg.sources:
        path = cfg.dir / f"{spec.name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run scripts/convert_public.py --datasets {spec.name}")
        eval_cap = spec.eval_cap if spec.eval_cap is not None else cfg.eval_cap
        train = _Capped(spec.cap)
        evals = {s: _Capped(eval_cap) for s in EVAL_SPLITS}
        available = unfit = 0
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                split = row["split"]
                if split != "train" and split not in evals:
                    continue
                available += split == "train"
                # The fit filter applies to eval splits too: a truncated rating is noisy to score.
                if spec.fit_only and not fits(row):
                    unfit += split == "train"
                    continue
                key = stable_hash(cfg.seed, "pool", row["id"])
                (train if split == "train" else evals[split]).add(key, row["type"], line.rstrip("\n"))
        by_prim: dict[str, list[str]] = defaultdict(list)
        for prim, line in train.items():
            by_prim[prim].append(line)
        for prim, lines in by_prim.items():
            pools.train[(spec.name, prim)] = lines
        for split, capped in evals.items():
            lines = [line for _, line in capped.items()]
            if lines:
                pools.eval[(split, spec.name)] = lines
        pools.available[spec.name] = available
        pools.dropped_unfit[spec.name] = unfit
    return pools


# ---------------------------------------------------------------------------------------------------
# Step 2: epochs


@dataclass(frozen=True)
class Allocation:
    source: str
    primitive: str
    pool: int
    drawn: int

    @property
    def repeat(self) -> float:
        return self.drawn / self.pool if self.pool else 0.0


def _largest_remainder(total: int, shares: dict[Any, float]) -> dict[Any, int]:
    """Integer counts summing to ``total``, proportional to ``shares``."""
    s = sum(shares.values())
    if total == 0 or s == 0:
        return {k: 0 for k in shares}
    exact = {k: total * v / s for k, v in shares.items()}
    counts = {k: math.floor(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[: total - sum(counts.values())]:
        counts[k] += 1
    return counts


def allocate(cfg: MixConfig, pools: Pools) -> list[Allocation]:
    """How many records every (source, primitive) pool contributes to one epoch."""
    weights = {s.name: s.weight for s in cfg.sources}
    quotas = _largest_remainder(cfg.total, cfg.primitive_weights)
    out: list[Allocation] = []
    for prim in PRIM_TYPES:
        keys = [k for k in pools.train if k[1] == prim]
        quota = quotas.get(prim, 0)
        if quota and not keys:
            raise ValueError(f"primitive {prim!r} has weight {cfg.primitive_weights[prim]} but no training records")
        counts = _largest_remainder(quota, {k: len(pools.train[k]) * weights[k[0]] for k in keys})
        for k in keys:
            a = Allocation(k[0], prim, len(pools.train[k]), counts[k])
            if a.repeat > cfg.max_repeat + 1e-9:
                raise ValueError(
                    f"{a.source}/{prim}: {a.drawn} records from a pool of {a.pool} is a {a.repeat:.2f}× repeat, "
                    f"above max_repeat={cfg.max_repeat}; lower the weight of {prim!r}, lower total, or raise caps"
                )
            out.append(a)
    return out


def epoch_lines(cfg: MixConfig, pools: Pools, allocations: Iterable[Allocation], epoch: int) -> list[str]:
    """The shuffled training lines of one epoch. Subsampled pools draw a fresh subset every epoch;
    repeated pools contribute every record ⌊r⌋ times plus a fresh subset for the remainder."""
    rng = random.Random(stable_hash(cfg.seed, "epoch", epoch))
    lines: list[str] = []
    for a in allocations:
        pool = pools.train[(a.source, a.primitive)]
        full, rest = divmod(a.drawn, a.pool) if a.pool else (0, 0)
        lines.extend(pool * full)
        lines.extend(rng.sample(pool, rest))
    rng.shuffle(lines)
    return lines


def eval_lines(pools: Pools, split: str) -> list[str]:
    return [line for (s, _), lines in sorted(pools.eval.items()) if s == split for line in lines]


def iter_records(lines: Iterable[str]) -> Iterator[Record]:
    for line in lines:
        yield Record.model_validate_json(line)


# ---------------------------------------------------------------------------------------------------
# Report


def mix_report(cfg: MixConfig, pools: Pools, allocations: list[Allocation]) -> dict:
    by_prim = defaultdict(int)
    for a in allocations:
        by_prim[a.primitive] += a.drawn
    rows = [
        {
            "source": a.source, "primitive": a.primitive, "available": pools.available[a.source],
            "dropped_unfit": pools.dropped_unfit[a.source], "pool": a.pool, "drawn": a.drawn,
            "repeat": round(a.repeat, 3), "share_of_primitive": round(a.drawn / by_prim[a.primitive], 4) if by_prim[a.primitive] else 0.0,
            "share_of_total": round(a.drawn / cfg.total, 4),
        }
        for a in allocations
    ]
    evals = defaultdict(dict)
    for (split, source), lines in sorted(pools.eval.items()):
        evals[split][source] = len(lines)
    return {
        "name": cfg.name, "total": cfg.total, "primitive_weights": cfg.primitive_weights, "max_len": cfg.max_len,
        "primitives": {p: {"drawn": by_prim.get(p, 0), "pool": sum(a.pool for a in allocations if a.primitive == p)} for p in PRIM_TYPES},
        "train": rows, "eval": {s: dict(v) for s, v in evals.items()},
    }


def report_markdown(report: dict) -> str:
    out = [f"# Mix `{report['name']}`", "", f"{report['total']} training records per epoch, max_len {report['max_len']}.", ""]
    out += ["| Primitive | Weight | Drawn | Pool | Repeat |", "|---|---|---|---|---|"]
    for p, v in report["primitives"].items():
        rep = v["drawn"] / v["pool"] if v["pool"] else 0.0
        out.append(f"| {p} | {report['primitive_weights'].get(p, 0):.0%} | {v['drawn']} | {v['pool']} | {rep:.2f}× |")
    out += ["", "| Source | Primitive | Available | Dropped (unfit) | Pool | Drawn | Repeat | Share of primitive | Share of total |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(report["train"], key=lambda r: (r["primitive"], -r["drawn"])):
        out.append(
            f"| {r['source']} | {r['primitive']} | {r['available']} | {r['dropped_unfit']} | {r['pool']} | {r['drawn']} "
            f"| {r['repeat']:.2f}× | {r['share_of_primitive']:.1%} | {r['share_of_total']:.1%} |"
        )
    for split, sources in report["eval"].items():
        out += ["", f"**{split}:** {sum(sources.values())} records — " + ", ".join(f"{k} {v}" for k, v in sources.items())]
    return "\n".join(out) + "\n"
