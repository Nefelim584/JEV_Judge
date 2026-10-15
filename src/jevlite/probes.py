"""Black-box checks of a predictor (TODO Phase 2 for Laya, Phase 8 for our packed model).

- **Order sensitivity**: how much the probability of an option moves when the options are permuted.
  A model that reads options independently (pair mode) scores 0.
- **IIA**: does adding an option change the ratios between the existing ones? Under IIA the
  log-ratio ``log p_i − log p_j`` stays the same.
- **Nonsense inputs**: does confidence drop when the state is replaced by shuffled words or random
  characters? If not, confidence cannot drive escalation. A confident "no" on a Bool question about
  garbage is the right answer, so the result is also split by primitive and by the predicted label
  (``nonsense_by_primitive``); the problem is confidence in a content answer (Choice / Score, Bool "yes").
- **Harder IIA** (``iia_same_source``): the added option comes from another record of the same source
  (same label space or question style), a near-miss rather than an option from an unrelated task.
- **Shortcuts on Bool** (``shortcuts``, per source): the claim / question with an empty state
  (hypothesis-only) and with the state of another record of the same source (passage swap). With no
  evidence, P(yes) should fall; hypothesis-only accuracy above the majority-class rate means the label
  can be read from the claim alone (todo, Stage A review).
"""

from __future__ import annotations

import random
import string
from itertools import combinations
from typing import Sequence

import numpy as np

from .baselines.base import Predictor, entropy_confidence
from .data.unified import Record
from .encoding import serialize_state, shuffle_choice
from .schema import ChoiceQuestion, Question, State


def order_sensitivity(predictor: Predictor, state: State, question: ChoiceQuestion, n_perms: int = 8, seed: int = 0) -> dict:
    """Spread of each option's probability over random permutations, mapped back to the original order."""
    rng = random.Random(seed)
    base = np.asarray(predictor.predict(state, [question])[0], dtype=float)
    runs = [base]
    for _ in range(n_perms):
        shuffled, _, perm = shuffle_choice(question, 0, rng)
        p_new = np.asarray(predictor.predict(state, [shuffled])[0], dtype=float)
        p_orig = np.empty_like(p_new)
        p_orig[perm] = p_new  # new position i holds original option perm[i]
        runs.append(p_orig)
    probs = np.stack(runs)
    spread = probs.max(0) - probs.min(0)
    return {
        "n_options": len(question.options),
        "mean_spread": float(spread.mean()),
        "max_spread": float(spread.max()),
        "argmax_flip_rate": float((probs.argmax(1) != base.argmax()).mean()),
    }


def iia_violation(predictor: Predictor, state: State, question: ChoiceQuestion, extra_option: str) -> dict:
    """Mean |Δ log-ratio| between the original options after ``extra_option`` is appended."""
    if extra_option in question.options:
        raise ValueError("the extra option is already an option")
    before = np.asarray(predictor.predict(state, [question])[0], dtype=float)
    extended = question.model_copy(update={"options": [*question.options, extra_option]})
    after = np.asarray(predictor.predict(state, [extended])[0], dtype=float)
    delta = np.log(np.clip(after[: len(before)], 1e-12, 1)) - np.log(np.clip(before, 1e-12, 1))
    pairs = list(combinations(range(len(before)), 2))
    change = float(np.mean([abs(delta[i] - delta[j]) for i, j in pairs])) if pairs else 0.0
    return {"mean_abs_log_ratio_change": change, "extra_option_mass": float(after[-1])}


def _shuffled_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def _random_chars(n: int, rng: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits + "     "
    return "".join(rng.choice(alphabet) for _ in range(n))


def nonsense_confidence(predictor: Predictor, state: State, questions: Sequence[Question], seed: int = 0) -> dict:
    """Mean heuristic confidence on the real state vs shuffled words vs random characters."""
    rng = random.Random(seed)
    text = serialize_state(state)
    variants = {"real": state, "shuffled_words": _shuffled_words(text, rng), "random_chars": _random_chars(len(text), rng)}
    return {
        name: float(np.mean([entropy_confidence(p) for p in predictor.predict(s, questions)]))
        for name, s in variants.items()
    }


EMPTY_STATE = "(no text)"  # hypothesis-only: an empty state is not a valid request


def _label(p: np.ndarray, prim: str) -> str:
    return ("yes" if p[0] >= 0.5 else "no") if prim == "bool" else "content"


def nonsense_detail(predictor: Predictor, record: Record, seed: int = 0) -> dict:
    """Per variant (real / shuffled words / random characters): confidence and predicted label."""
    rng = random.Random(seed)
    text = serialize_state(record.state)
    variants = {"real": record.state, "shuffled_words": _shuffled_words(text, rng), "random_chars": _random_chars(len(text), rng)}
    q = record.to_question()
    out = {"primitive": record.type}
    for name, st in variants.items():
        p = np.asarray(predictor.predict(st, [q])[0], dtype=float)
        out[name] = {"confidence": float(entropy_confidence(p)), "label": _label(p, record.type)}
    return out


def nonsense_breakdown(rows: Sequence[dict]) -> dict:
    """Mean confidence per primitive and predicted label on each variant."""
    out: dict = {}
    for prim in sorted({r["primitive"] for r in rows}):
        sub = [r for r in rows if r["primitive"] == prim]
        cell: dict = {"n": len(sub)}
        for variant in ("real", "shuffled_words", "random_chars"):
            by_label: dict = {}
            for lab in sorted({r[variant]["label"] for r in sub}):
                confs = [r[variant]["confidence"] for r in sub if r[variant]["label"] == lab]
                by_label[lab] = {"n": len(confs), "confidence": float(np.mean(confs))}
            cell[variant] = {"confidence": float(np.mean([r[variant]["confidence"] for r in sub])), "by_label": by_label}
        out[prim] = cell
    return out


def shortcut_probe(predictor: Predictor, records: Sequence[Record], n_per_source: int = 200, seed: int = 0) -> dict:
    """Hypothesis-only and passage-swap checks on the Bool records of every source."""
    out: dict = {}
    by_source: dict[str, list[Record]] = {}
    for r in records:
        if r.type == "bool":
            by_source.setdefault(r.source, []).append(r)
    for source, recs in sorted(by_source.items()):
        rng = random.Random(f"{seed}:{source}")
        sample = rng.sample(recs, min(n_per_source, len(recs)))
        if len(sample) < 2:
            continue
        y = np.asarray([float(r.target) >= 0.5 for r in sample])
        p = {"real": [], "hyp_only": [], "swap": []}
        for i, r in enumerate(sample):
            q = r.to_question()
            other = sample[(i + 1 + rng.randrange(len(sample) - 1)) % len(sample)]  # never the record itself
            for name, st in (("real", r.state), ("hyp_only", EMPTY_STATE), ("swap", other.state)):
                p[name].append(float(predictor.predict(st, [q])[0][0]))
        row = {"n": len(sample), "positive_rate": float(y.mean()), "majority_rate": float(max(y.mean(), 1 - y.mean()))}
        for name, ps in p.items():
            ps = np.asarray(ps)
            row[name] = {"accuracy": float(((ps >= 0.5) == y).mean()), "p_yes": float(ps.mean()), "share_yes": float((ps >= 0.5).mean())}
        row["hyp_only_above_majority"] = row["hyp_only"]["accuracy"] - row["majority_rate"]
        out[source] = row
    return out


def run_probes(
    predictor: Predictor, records: Sequence[Record], n_items: int = 50, n_perms: int = 8, seed: int = 0,
    shortcuts_per_source: int = 0,
) -> dict:
    """All checks on up to ``n_items`` records. IIA adds an option taken from another record (any
    source, and in ``iia_same_source`` from the same source); ``shortcuts_per_source > 0`` adds the
    Bool shortcut probe."""
    rng = random.Random(seed)
    choice = [r for r in records if r.type == "choice" and len(r.candidates) >= 2]
    rng.shuffle(choice)
    choice = choice[:n_items]
    pool = sorted({c for r in records if r.candidates for c in r.candidates})

    by_source = {}
    for r in records:
        if r.candidates:
            by_source.setdefault(r.source, set()).update(r.candidates)

    order, iia, iia_same = [], [], []
    for i, r in enumerate(choice):
        q = r.to_question()
        order.append(order_sensitivity(predictor, r.state, q, n_perms, seed + i))
        extra = [c for c in pool if c not in q.options]
        if extra:
            iia.append(iia_violation(predictor, r.state, q, rng.choice(extra)))
        near = sorted(c for c in by_source.get(r.source, ()) if c not in q.options)
        if near:
            iia_same.append(iia_violation(predictor, r.state, q, random.Random(f"{seed}:near:{r.id}").choice(near)))

    sample = list(records)
    rng.shuffle(sample)
    detail = [nonsense_detail(predictor, r, seed + i) for i, r in enumerate(sample[:n_items])]
    nonsense = [{k: d[k]["confidence"] for k in ("real", "shuffled_words", "random_chars")} for d in detail]

    def _mean(rows: list[dict], key: str) -> float | None:
        return float(np.mean([row[key] for row in rows])) if rows else None

    return {
        "predictor": predictor.name,
        "order_sensitivity": {
            "n": len(order),
            "mean_spread": _mean(order, "mean_spread"),
            "max_spread": _mean(order, "max_spread"),
            "argmax_flip_rate": _mean(order, "argmax_flip_rate"),
        },
        "iia": {
            "n": len(iia),
            "mean_abs_log_ratio_change": _mean(iia, "mean_abs_log_ratio_change"),
            "extra_option_mass": _mean(iia, "extra_option_mass"),
        },
        "iia_same_source": {
            "n": len(iia_same),
            "mean_abs_log_ratio_change": _mean(iia_same, "mean_abs_log_ratio_change"),
            "extra_option_mass": _mean(iia_same, "extra_option_mass"),
        },
        "nonsense_confidence": {
            "n": len(nonsense),
            "real": _mean(nonsense, "real"),
            "shuffled_words": _mean(nonsense, "shuffled_words"),
            "random_chars": _mean(nonsense, "random_chars"),
        },
        "nonsense_by_primitive": nonsense_breakdown(detail),
        **({"shortcuts": shortcut_probe(predictor, records, shortcuts_per_source, seed)} if shortcuts_per_source else {}),
    }
