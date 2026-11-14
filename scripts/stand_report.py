"""Agreement of jev-lite with the production judge on the eval stand (Stage C test, ``jevlite.data.stand``).

    uv run python scripts/stand_report.py --records data/stand/records/stand.jsonl \\
        --pred stage_c=data/runs/stage_c-full-s42/preds.jsonl --out data/runs/stage_c-full-s42/stand

Per Score criterion of the stand (names come from the records, never from the code), on ``test_in``:

- **exact**: the model's most likely level = the judge's score (for judge repeats, S4: their mode);
  **within ±1**; **bucket**: the same of ≥4 / 3 / ≤2; MAE; quadratic-weighted kappa; Spearman of the
  expected level; NLL. 95% bootstrap CIs over stand rows.
- the **constant baseline**: always the most frequent level of the stand's train split.
- **S4** (rows the judge scored three times): the model against each single judge call (the same
  question as "how often would a second judge call agree?") next to judge-vs-judge agreement, the ceiling.
- the weak faithfulness Bool: accuracy and AUROC.
- slices by category family (the part of ``category`` before the first ``_``) and single / multi turn.

Writes ``<out>/stand_report.json`` and ``.md``: numbers only, no texts. The records are company data:
keep them and the output under the git-ignored ``data/``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from jevlite.data.unified import read_jsonl


def _named(spec: str) -> tuple[str, str]:
    name, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, path


def qwk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Quadratic-weighted Cohen's kappa of two integer label arrays in 0..k-1."""
    if len(a) == 0:
        return float("nan")
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def bucket(level: np.ndarray) -> np.ndarray:
    """0-based level of 5 → 0 (≤2), 1 (3), 2 (≥4)."""
    return np.where(level <= 1, 0, np.where(level == 2, 1, 2))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # ties: average ranks
    allp = np.concatenate([pos, neg])
    for v in np.unique(allp):
        m = allp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(fn, n: int, reps: int, seed: int = 0) -> list[float]:
    if n < 2 or reps <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    vals = [v for v in vals if not math.isnan(v)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def score_metrics(target: np.ndarray, probs: np.ndarray, reps: int) -> dict:
    """``target``: 0-based levels; ``probs``: (n, k)."""
    n, k = probs.shape
    pred = probs.argmax(1)
    expv = probs @ np.arange(k)
    nll = -np.log(np.clip(probs[np.arange(n), target], 1e-12, None))
    m = {
        "n": int(n),
        "exact": float((pred == target).mean()),
        "within1": float((np.abs(pred - target) <= 1).mean()),
        "bucket": float((bucket(pred) == bucket(target)).mean()),
        "mae": float(np.abs(pred - target).mean()),
        "qwk": qwk(pred, target, k),
        "spearman": spearman(expv, target.astype(float)),
        "nll": float(nll.mean()),
        "pred_dist": {str(i + 1): int(c) for i, c in sorted(Counter(pred.tolist()).items())},
        "target_dist": {str(i + 1): int(c) for i, c in sorted(Counter(target.tolist()).items())},
    }
    m["exact_ci"] = boot_ci(lambda ix: float((pred[ix] == target[ix]).mean()), n, reps)
    m["within1_ci"] = boot_ci(lambda ix: float((np.abs(pred[ix] - target[ix]) <= 1).mean()), n, reps, 1)
    m["qwk_ci"] = boot_ci(lambda ix: qwk(pred[ix], target[ix], k), n, reps, 2)
    return m


def constant_metrics(target: np.ndarray, level: int, k: int) -> dict:
    pred = np.full_like(target, level)
    return {"level": level + 1, "exact": float((pred == target).mean()),
            "within1": float((np.abs(pred - target) <= 1).mean()),
            "bucket": float((bucket(pred) == bucket(target)).mean()), "mae": float(np.abs(pred - target).mean())}


def s4_metrics(rows: list[dict], probs: dict[str, np.ndarray]) -> dict:
    """Rows with ≥2 judge scores: model vs each judge call, judge vs judge (all pairs), model vs mode."""
    m_exact, m_w1, j_exact, j_w1, m_mode = [], [], [], [], []
    for r in rows:
        js = [int(s) - 1 for s in r["judge_scores"]]
        p = int(probs[r["id"]].argmax())
        m_exact += [p == j for j in js]
        m_w1 += [abs(p - j) <= 1 for j in js]
        for a, b in combinations(js, 2):
            j_exact.append(a == b)
            j_w1.append(abs(a - b) <= 1)
        m_mode.append(p == int(r["target"]))
    f = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {"rows": len(rows), "model_vs_single_call_exact": f(m_exact), "model_vs_single_call_within1": f(m_w1),
            "judge_vs_judge_exact": f(j_exact), "judge_vs_judge_within1": f(j_w1), "model_vs_mode_exact": f(m_mode)}


def family(category: str | None) -> str:
    c = (category or "unknown").lower()
    return c.split("_")[0] if c.startswith("attack") else c


def report(records: list[dict], preds: dict[str, dict[str, dict]], reps: int) -> dict:
    test = [r for r in records if r.get("split") == "test_in"]
    train = [r for r in records if r.get("split") == "train"]
    criteria = sorted({r["criterion"] for r in test if r["type"] == "score"})
    out: dict = {"criteria": criteria, "test_rows": len({r["stand_row"] for r in test}), "models": {}, "constant": {}}

    for c in criteria:
        tr = [int(np.argmax(r["target"])) if isinstance(r["target"], list) else int(r["target"]) for r in train
              if r["criterion"] == c and r["type"] == "score"]
        level = Counter(tr).most_common(1)[0][0]
        t = np.array([int(r["target"]) for r in test if r["criterion"] == c and r["type"] == "score"])
        out["constant"][c] = constant_metrics(t, level, 5)

    for name, pmap in preds.items():
        res: dict = {"by_criterion": {}, "s4": {}, "slices": {}, "faithfulness": None}
        all_t, all_p = [], []
        for c in criteria:
            rows = [r for r in test if r["criterion"] == c and r["type"] == "score" and r["id"] in pmap]
            if not rows:
                continue
            t = np.array([int(r["target"]) for r in rows])
            p = np.array([pmap[r["id"]]["probs"] for r in rows], dtype=float)
            res["by_criterion"][c] = score_metrics(t, p, reps)
            all_t.append(t)
            all_p.append(p)
            s4 = [r for r in rows if len(r.get("judge_scores") or []) >= 2]
            if s4:
                res["s4"][c] = s4_metrics(s4, {r["id"]: np.array(pmap[r["id"]]["probs"]) for r in s4})
            by: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(rows):
                by[f"family={family(r.get('category'))}"].append(i)
                by[f"turns={'multi' if r.get('multi_turn') else 'single'}"].append(i)
            res["slices"][c] = {k: {"n": len(ix), "exact": float((p[ix].argmax(1) == t[ix]).mean()),
                                    "within1": float((np.abs(p[ix].argmax(1) - t[ix]) <= 1).mean())}
                                for k, ix in sorted(by.items())}
        if all_t:
            res["overall"] = score_metrics(np.concatenate(all_t), np.concatenate(all_p), reps)
        fb = [r for r in test if r["type"] == "bool" and r["id"] in pmap]
        if fb:
            y = np.array([float(r["target"]) for r in fb])
            pr = np.array([float(pmap[r["id"]]["prob"]) for r in fb])
            res["faithfulness"] = {"n": len(fb), "positives": int((y >= 0.5).sum()),
                                   "accuracy": float(((pr >= 0.5) == (y >= 0.5)).mean()), "auroc": auroc((y >= 0.5).astype(int), pr)}
        res["missing"] = sum(1 for r in test if r["id"] not in pmap)
        out["models"][name] = res
    return out


def _pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def to_markdown(rep: dict) -> str:
    L = [f"# Eval stand: agreement with the production judge (test_in, {rep['test_rows']} rows)", "",
         "Exact = the model's most likely level equals the judge's score (S4 rows: the mode of 3 judge calls). "
         "Bucket = same of ≥4 / 3 / ≤2. CIs: 95% bootstrap over rows. Constant = the most frequent train level.", ""]
    for name, res in rep["models"].items():
        L += [f"## {name}", "", "| Criterion | n | Exact [CI] | ±1 [CI] | Bucket | MAE | QWK [CI] | Spearman | NLL | Constant exact (level) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c, m in res["by_criterion"].items():
            k = rep["constant"][c]
            L.append(f"| {c} | {m['n']} | {_pct(m['exact'])} [{_pct(m['exact_ci'][0])}, {_pct(m['exact_ci'][1])}] | "
                     f"{_pct(m['within1'])} [{_pct(m['within1_ci'][0])}, {_pct(m['within1_ci'][1])}] | {_pct(m['bucket'])} | "
                     f"{m['mae']:.2f} | {m['qwk']:.2f} [{m['qwk_ci'][0]:.2f}, {m['qwk_ci'][1]:.2f}] | {m['spearman']:.2f} | "
                     f"{m['nll']:.2f} | {_pct(k['exact'])} ({k['level']}) |")
        if "overall" in res:
            m = res["overall"]
            L.append(f"| **all** | {m['n']} | **{_pct(m['exact'])}** [{_pct(m['exact_ci'][0])}, {_pct(m['exact_ci'][1])}] | "
                     f"**{_pct(m['within1'])}** | {_pct(m['bucket'])} | {m['mae']:.2f} | {m['qwk']:.2f} | {m['spearman']:.2f} | {m['nll']:.2f} | |")
        if res["s4"]:
            L += ["", "**S4: the judge scored these rows 3 times.** Model vs one judge call, next to judge vs judge (the ceiling).", "",
                  "| Criterion | rows | Model–judge exact | Judge–judge exact | Model–judge ±1 | Judge–judge ±1 | Model vs mode |",
                  "|---|---|---|---|---|---|---|"]
            for c, s in res["s4"].items():
                L.append(f"| {c} | {s['rows']} | {_pct(s['model_vs_single_call_exact'])} | {_pct(s['judge_vs_judge_exact'])} | "
                         f"{_pct(s['model_vs_single_call_within1'])} | {_pct(s['judge_vs_judge_within1'])} | {_pct(s['model_vs_mode_exact'])} |")
        if res["faithfulness"]:
            f = res["faithfulness"]
            L += ["", f"Weak faithfulness Bool (from the judge's accuracy score): n {f['n']} ({f['positives']} positive), "
                      f"accuracy {_pct(f['accuracy'])}, AUROC {f['auroc']:.3f}."]
        L += ["", "**Slices** (exact / ±1):", "", "| Criterion | " + " | ".join(next(iter(res["slices"].values())).keys()) + " |",
              "|---|" + "---|" * len(next(iter(res["slices"].values())))]
        for c, sl in res["slices"].items():
            L.append(f"| {c} | " + " | ".join(f"{_pct(v['exact'])} / {_pct(v['within1'])} (n {v['n']})" for v in sl.values()) + " |")
        L += ["", "Level distribution, judge → model:", ""]
        for c, m in res["by_criterion"].items():
            L.append(f"- {c}: judge {m['target_dist']} → model {m['pred_dist']}")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="stand records (convert_stand.py output) or exported labels")
    ap.add_argument("--pred", type=_named, action="append", required=True, help="NAME=preds.jsonl (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)

    records = [r for r in read_jsonl(args.records) if r.get("source", "stand") == "stand"]
    preds = {name: {p["id"]: p for p in read_jsonl(path)} for name, path in args.pred}
    rep = report(records, preds, args.bootstrap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stand_report.json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    (out / "stand_report.md").write_text(md)
    print(md)
    return rep


if __name__ == "__main__":
    main()
