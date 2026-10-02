"""Black-box checks of a predictor (TODO Phase 2 for Laya, Phase 10 for our packed model).

- **Order sensitivity**: how much the probability of an option moves when the options are permuted.
  A model that reads options independently (pair mode) scores 0.
- **IIA**: does adding an option change the ratios between the existing ones? Under IIA the
  log-ratio ``log p_i − log p_j`` stays the same.
- **Nonsense inputs**: does confidence drop when the state is replaced by shuffled words or random
  characters? If not, confidence cannot drive escalation.
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


def run_probes(predictor: Predictor, records: Sequence[Record], n_items: int = 50, n_perms: int = 8, seed: int = 0) -> dict:
    """All three checks on up to ``n_items`` records. IIA adds an option taken from another record."""
    rng = random.Random(seed)
    choice = [r for r in records if r.type == "choice" and len(r.candidates) >= 2]
    rng.shuffle(choice)
    choice = choice[:n_items]
    pool = sorted({c for r in records if r.candidates for c in r.candidates})

    order, iia = [], []
    for i, r in enumerate(choice):
        q = r.to_question()
        order.append(order_sensitivity(predictor, r.state, q, n_perms, seed + i))
        extra = [c for c in pool if c not in q.options]
        if extra:
            iia.append(iia_violation(predictor, r.state, q, rng.choice(extra)))

    sample = list(records)
    rng.shuffle(sample)
    nonsense = [nonsense_confidence(predictor, r.state, [r.to_question()], seed + i) for i, r in enumerate(sample[:n_items])]

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
        "nonsense_confidence": {
            "n": len(nonsense),
            "real": _mean(nonsense, "real"),
            "shuffled_words": _mean(nonsense, "shuffled_words"),
            "random_chars": _mean(nonsense, "random_chars"),
        },
    }
