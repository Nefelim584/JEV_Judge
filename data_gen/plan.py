"""Case plans: every attribute of a case drawn by code from the quotas in ``config.yaml``.

The model never decides what kind of case it writes. Diversity comes from the plan (domain × topic ×
aspect × formats × question kind × answer kinds) and from the topic taxonomy, which is generated once,
deduplicated and frozen before planning.
"""

from __future__ import annotations

import random
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def draw(rng: random.Random, dist: dict) -> Any:
    keys = list(dist)
    return rng.choices(keys, weights=[dist[k] for k in keys])[0]


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def heldout_style(kind: str, styles: list[str]) -> str:
    """One question style per kind is held out for test_ood (dataset_dicr.md 5, 12)."""
    return sorted(styles)[zlib.crc32(kind.encode()) % len(styles)]


def _split(rng: random.Random, cfg: dict, domain: str) -> str:
    if cfg["domains"][domain].get("ood"):
        return "test_ood"
    return draw(rng, cfg["splits"])


def _paraphrase(rng: random.Random, cfg: dict, difficulty: str) -> str:
    if difficulty == "easy":
        return rng.choice(["verbatim", "light"])
    if difficulty == "hard":
        return draw(rng, {"strong": 0.7, "light": 0.3})
    return draw(rng, cfg["answer"]["paraphrase"])


def _perturbations(rng: random.Random, cfg: dict, split: str, citations: bool) -> list[str]:
    dist = dict(cfg["perturbation"]["kind"])
    if split != "test_ood":
        for k in cfg["perturbation"]["heldout"]:
            dist.pop(k, None)
    if not citations:
        dist.pop("citation_swap", None)
    n = draw(rng, cfg["answer"]["perturbations_per_answer"])
    kinds: list[str] = []
    while len(kinds) < n:
        k = draw(rng, dist)
        if k not in kinds:
            kinds.append(k)
        if k == "citation_swap":      # the text stays the same: nothing else goes with it
            return ["citation_swap"]
    return kinds


def _answers(rng: random.Random, cfg: dict, split: str, answerable: bool, question_kind: str,
             citations: bool) -> list[dict]:
    a = cfg["answer"]
    if not answerable:
        answers = [{"kind": "hallucinated"}, {"kind": "refusal_justified"}]
        extra = draw(rng, a["extra_unanswerable"])
    else:
        answers = [{"kind": "grounded"}]
        for _ in range(draw(rng, a["n_perturbed"])):
            answers.append({"kind": "perturbed", "perturbations": _perturbations(rng, cfg, split, citations)})
        extra = draw(rng, a["extra_answerable"])
        if extra == "partial_refusal" and question_kind not in ("multi_part", "comparison", "list"):
            extra = "none"            # a partial refusal needs a question with parts
    if extra != "none":
        answers.append({"kind": extra})
    for i, ans in enumerate(answers):
        ans["answer_id"] = f"a{i}"
    return answers


class TopicPicker:
    """Least-used topic of the domain, then least-used aspect of that topic (random tie-breaks)."""

    def __init__(self, topics: list[dict]):
        self.by_domain: dict[str, list[dict]] = {}
        for t in topics:
            self.by_domain.setdefault(t["domain"], []).append(t)
        self.used: Counter = Counter()

    def pick(self, rng: random.Random, domain: str) -> tuple[dict, str]:
        topics = self.by_domain.get(domain)
        if not topics:
            raise ValueError(f"no topics for domain {domain}: run the taxonomy step first")
        low = min(self.used[t["topic_id"]] for t in topics)
        topic = rng.choice([t for t in topics if self.used[t["topic_id"]] == low])
        aspects = topic.get("aspects") or ["overview"]
        low_a = min(self.used[(topic["topic_id"], a)] for a in aspects)
        aspect = rng.choice([a for a in aspects if self.used[(topic["topic_id"], a)] == low_a])
        self.used[topic["topic_id"]] += 1
        self.used[(topic["topic_id"], aspect)] += 1
        return topic, aspect


def make_plan(cfg: dict, idx: int, picker: TopicPicker) -> dict:
    rng = stable_rng(cfg["seed"], "plan", idx)
    ch, q, a = cfg["chunks"], cfg["question"], cfg["answer"]
    domain = draw(rng, {d: v["weight"] for d, v in cfg["domains"].items()})
    split = _split(rng, cfg, domain)
    topic, aspect = picker.pick(rng, domain)
    difficulty = draw(rng, cfg["difficulty"])
    answerable = rng.random() >= ch["unanswerable"]

    kind = draw(rng, q["kind"])
    styles = list(q["styles"])
    held = heldout_style(kind, styles)
    if split == "test_ood" and not cfg["domains"][domain].get("ood"):
        style = held                                   # the template axis of test_ood
    else:
        style = rng.choice([s for s in styles if s != held])

    n_chunks = draw(rng, ch["n_chunks"])
    n_relevant = 2 if kind in q["two_chunk_kinds"] or rng.random() < 0.25 else 1
    near_miss = difficulty == "hard" or rng.random() < ch["near_miss"]
    conflict = answerable and rng.random() < ch["conflict"]
    near_dup = rng.random() < ch["near_duplicate"]
    # the extras must leave room for the relevant sections: drop them in this order
    for flag in ("near_dup", "near_miss", "conflict"):
        if n_relevant + near_miss + conflict + near_dup <= n_chunks:
            break
        if flag == "near_dup":
            near_dup = False
        elif flag == "near_miss":
            near_miss = False
        else:
            conflict = False
    n_relevant = min(n_relevant, n_chunks)
    extras = near_miss + conflict + near_dup
    n_random_target = 0 if n_chunks <= 2 else 1 if n_chunks <= 4 else 2
    n_doc = max(n_relevant, n_chunks - extras - n_random_target)
    n_random = max(0, n_chunks - extras - n_doc)

    sections = [{"format": draw(rng, ch["format"]), "words": draw(rng, ch["words"])} for _ in range(n_doc)]
    relevant = sorted(rng.sample(range(n_doc), n_relevant))

    citations = rng.random() < a["citations"]
    lo, hi = map(int, draw(rng, a["n_sentences"]).split("-"))
    plan = {
        "case_id": f"{cfg['dataset_version']}-{idx:06d}",
        "idx": idx,
        "seed": rng.randrange(2**31),
        "split": split,
        "domain": domain,
        "topic_id": topic["topic_id"],
        "topic": topic["name"],
        "topic_summary": topic.get("summary", ""),
        "aspect": aspect,
        "real_world": bool(topic.get("real")),
        "difficulty": difficulty,
        "answerable": answerable,
        "n_chunks": n_chunks,
        "sections": sections,
        "relevant_sections": relevant,
        "near_miss": near_miss,
        "conflict": conflict,
        "near_duplicate": near_dup,
        "n_random": n_random,
        "question": {"kind": kind, "style": style, "template_id": f"q.{kind}.{style}"},
        "answer_style": {
            "form": draw(rng, a["form"]),
            "n_sentences": [lo, hi],
            "citations": citations,
            "nonfactual": rng.random() < a["nonfactual"],
            "hedging": rng.random() < a["hedging"],
            "paraphrase": _paraphrase(rng, cfg, difficulty),
            "multi_chunk_sentence": n_relevant >= 2 and (kind == "comparison" or rng.random() < 0.3),
        },
        "answers": _answers(rng, cfg, split, answerable, kind, citations),
    }
    return plan


def make_plans(cfg: dict, topics: list[dict], n: int, start: int = 0, used: list[dict] | None = None) -> list[dict]:
    """Plans for cases ``start … start+n-1``. ``used`` are earlier plans, so topic use stays balanced."""
    picker = TopicPicker(topics)
    for p in used or []:
        picker.used[p["topic_id"]] += 1
        picker.used[(p["topic_id"], p["aspect"])] += 1
    return [make_plan(cfg, i, picker) for i in range(start, start + n)]
