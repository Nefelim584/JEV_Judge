"""JEV (TypeSafe AI) over its API, on a stratified sample of the clean Stage A test sets, for a paired
comparison with jev-lite, NLI and Laya (``scripts/compare_paired.py``).

Needs ``typesafe-sdk`` (``uv add typesafe-sdk``) and ``TYPESAFE_API_KEY`` in ``.env``.

    # 1. Sample (no API calls): 700 records per (split, primitive), sources in proportion; prints calls and tokens
    uv run python scripts/predict_jev_api.py sample --out data/jev_api/sample.jsonl

    # 2. Check the model name and one request without sending it
    uv run python scripts/predict_jev_api.py models
    uv run python scripts/predict_jev_api.py predict --data data/jev_api/sample.jsonl --out data/jev_api/preds.jsonl --dry-run

    # 3. Pilot: the first 200 records of the sample (it is shuffled, so they span every cell), then the rest
    uv run python scripts/predict_jev_api.py predict --data data/jev_api/sample.jsonl --out data/jev_api/preds.jsonl --limit 200
    uv run python scripts/predict_jev_api.py repeat  --data data/jev_api/sample.jsonl --preds data/jev_api/preds.jsonl --n 30
    uv run python scripts/predict_jev_api.py predict --data data/jev_api/sample.jsonl --out data/jev_api/preds.jsonl --resume

    # 4. Paired comparison on the sampled records (records without a JEV prediction are left out)
    uv run python scripts/compare_paired.py --data data/jev_api/sample.jsonl --model jevlite=data/runs/stage_a-full-s42/preds.jsonl \\
        --ref jev=data/jev_api/preds.jsonl --ref nli=data/baselines/preds/nli.jsonl --ref laya=data/baselines/preds/laya.jsonl \\
        --out data/jev_api/paired

Round 2 (STAGE_A_API.md section 6):

    # more records per source (no API calls), then predict them into the same file
    uv run python scripts/predict_jev_api.py topup
    uv run python scripts/predict_jev_api.py predict --data data/jev_api/sample.jsonl data/jev_api/sample_topup.jsonl \\
        --out data/jev_api/preds.jsonl --resume
    # Bool wording check: the first 700 Bool records of the sample, phrased as a question / with yes-no criteria
    uv run python scripts/predict_jev_api.py predict --data data/jev_api/sample.jsonl --types bool --limit 700 \\
        --bool-style question --out data/jev_api/preds_bool_question.jsonl
    # black-box probes, the same data, items and seed as for NLI, Laya and jev-lite
    uv run python scripts/predict_jev_api.py probes --data data/mix/stage_a/clean/test_in.jsonl --out data/probes_2026-10-11/jev.json

One call per record. Mapping of our primitives, as for Laya: Choice → ``Choice`` with the options as
criteria, Score → ``Score`` with the levels in order, Bool → ``Noul`` (P(yes) = ``noul``). ``confidence``
is the entropy heuristic used for every baseline; JEV's own confidence is kept as ``api_confidence``.
Each prediction also carries ``model``, ``input_tokens``, ``output_tokens`` and ``request_id``.
Predictions are appended as they come; ``--resume`` skips the records already in ``--out``, failed
records go to ``<out>.skipped.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from dotenv import load_dotenv

from jevlite.baselines.base import entropy_confidence
from jevlite.data.unified import Prediction, Record, load_predictions, load_records
from jevlite.encoding import serialize_state
from jevlite.schema import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion, State

CLEAN = "data/mix/stage_a/clean"
DEFAULT_DATA = [f"{CLEAN}/test_in.jsonl", f"{CLEAN}/test_ood_source.jsonl", f"{CLEAN}/test_ood_template.jsonl"]
DEFAULT_MODEL = "jev-latest"
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------- sample


def _allocate(counts: Counter, n: int) -> dict[str, int]:
    """``n`` split over sources in proportion to ``counts`` (largest remainder), capped by availability."""
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    exact = {s: n * c / total for s, c in counts.items()}
    alloc = {s: int(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: exact[s] - alloc[s], reverse=True)[: n - sum(alloc.values())]:
        alloc[s] += 1
    return alloc


def _est_tokens(r: Record) -> int:
    text = r.state if isinstance(r.state, str) else json.dumps(r.state, ensure_ascii=False)
    return (len(text) + len(r.prompt) + sum(len(c) for c in r.candidates or [])) // CHARS_PER_TOKEN


def cmd_sample(args) -> None:
    records = [r for path in args.data for r in load_records(path)]
    cells: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        cells[(r.split, r.type)].append(r)
    rng = random.Random(args.seed)
    sample: list[Record] = []
    print(f"| split | primitive | available | sampled | est. input tokens |\n|---|---|---|---|---|")
    for (split, prim), recs in sorted(cells.items()):
        by_source: dict[str, list[Record]] = defaultdict(list)
        for r in recs:
            by_source[r.source].append(r)
        alloc = _allocate(Counter({s: len(v) for s, v in by_source.items()}), args.per_cell)
        picked = [r for s in sorted(alloc) for r in rng.sample(sorted(by_source[s], key=lambda r: r.id), alloc[s])]
        sample += picked
        print(f"| {split} | {prim} | {len(recs):,} | {len(picked):,} | {sum(map(_est_tokens, picked)):,} |")
    rng.shuffle(sample)  # so that any prefix (--limit) spans every cell
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in sample:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    print(f"\n{len(sample):,} records = {len(sample):,} API calls, ~{sum(map(_est_tokens, sample)):,} input tokens "
          f"(chars / {CHARS_PER_TOKEN}) → {out}")


# ---------------------------------------------------------------------------- API


def cmd_topup(args) -> None:
    """More records per (source, primitive) on top of an existing sample: trained sources (``test_in`` +
    ``test_ood_template`` pooled) up to ``--per-source``, new sources (``test_ood_source``) up to
    ``--per-new-source``; records already in ``--exclude`` count towards the target and are not drawn again."""
    taken = {r.id for path in args.exclude for r in load_records(path)}
    groups: dict[tuple[str, str, bool], list[Record]] = defaultdict(list)
    for r in (r for path in args.data for r in load_records(path)):
        groups[(r.source, r.type, r.split == "test_ood_source")].append(r)
    rng = random.Random(args.seed)
    sample: list[Record] = []
    print("| source | primitive | new source | available | already | added |\n|---|---|---|---|---|---|")
    for (source, prim, new), recs in sorted(groups.items()):
        have = sum(r.id in taken for r in recs)
        free = sorted((r for r in recs if r.id not in taken), key=lambda r: r.id)
        n = max(0, min(args.per_new_source if new else args.per_source, len(recs)) - have)
        picked = rng.sample(free, min(n, len(free)))
        sample += picked
        print(f"| {source} | {prim} | {'yes' if new else ''} | {len(recs):,} | {have:,} | {len(picked):,} |")
    rng.shuffle(sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in sample:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    print(f"\n{len(sample):,} records = {len(sample):,} API calls, ~{sum(map(_est_tokens, sample)):,} input tokens "
          f"(chars / {CHARS_PER_TOKEN}) → {out}")


# How a Bool statement is put to JEV's ``Noul``. "statement" (the default, as for Laya) sends it unchanged.
BOOL_STYLES = ("statement", "question", "criteria")
BOOL_CRITERIA = {"true": "The statement is true according to the text.",
                 "false": "The statement is false, or the text does not support it."}


def _jev_question(q: Question, bool_style: str = "statement"):
    """One of our questions (``jevlite.schema``) → a JEV question."""
    from typesafe_sdk import Choice, Noul, Score

    if isinstance(q, ChoiceQuestion):
        return Choice(instructions=q.prompt, criteria={o: None for o in q.options})
    if isinstance(q, ScoreQuestion):
        return Score(instructions=q.prompt, criteria=list(q.levels))
    if bool_style == "question":
        return Noul(instructions=f"Based on the text, is the following statement true? {q.prompt}")
    if bool_style == "criteria":
        return Noul(instructions=q.prompt, criteria=BOOL_CRITERIA)
    return Noul(instructions=q.prompt)


def _question(r: Record, bool_style: str = "statement"):
    return _jev_question(r.to_question(qid="q"), bool_style)


def _probs(q: Question, answer) -> np.ndarray:
    """JEV's answer → our layout: probabilities in option / level order, ``[P(yes)]`` for Bool."""
    if isinstance(q, BoolQuestion):
        return np.array([float(np.clip(answer.noul, 0.0, 1.0))])
    if isinstance(q, ChoiceQuestion):
        got = answer.probabilities
        if set(got) != set(q.options):
            raise ValueError(f"returned options differ from the question's: {sorted(got)[:5]}")
        p = np.array([got[o] for o in q.options], dtype=float)
    else:
        got = {int(k): v for k, v in answer.probabilities.items()}
        p = np.array([got.get(i, 0.0) for i in range(len(q.levels))], dtype=float)
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def _to_prediction(r: Record, response, latency_ms: float) -> Prediction:
    answer = response.answers["q"]
    extra = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens, "request_id": response.request_id}
    p = _probs(r.to_question(qid="q"), answer)
    if r.type == "bool":
        return Prediction(id=r.id, prob=float(p[0]), confidence=entropy_confidence(p), latency_ms=latency_ms, **extra)
    return Prediction(id=r.id, probs=p.tolist(), confidence=entropy_confidence(p), latency_ms=latency_ms,
                      api_confidence=float(answer.confidence), **extra)


def _api_key(args) -> str:
    load_dotenv()
    key = os.environ.get(args.key_env, "").strip()
    if not key:
        raise SystemExit(f"{args.key_env} is not set (add it to .env)")
    return key


def _client(args):
    try:
        from typesafe_sdk import AsyncTypeSafeClient
    except ImportError:
        raise SystemExit("typesafe-sdk is not installed: uv add typesafe-sdk")
    return AsyncTypeSafeClient(api_key=_api_key(args), model=args.model, timeout=args.timeout)


async def _predict_one(client, r: Record, sem: asyncio.Semaphore, bool_style: str = "statement") -> Prediction:
    async with sem:
        t0 = time.perf_counter()
        response = await client.system_one(state=r.state, questions={"q": _question(r, bool_style)})
        return _to_prediction(r, response, (time.perf_counter() - t0) * 1000)


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


async def _run(args, records: list[Record], out: Path, skipped_path: Path) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    n_pred = n_skip = tok_in = tok_out = 0
    start = time.perf_counter()
    async with _client(args) as client:
        async def task(r: Record):
            try:
                return r, await _predict_one(client, r, sem, args.bool_style), None
            except Exception as e:  # one bad record must not stop a paid run
                return r, None, f"{type(e).__name__}: {e}"

        with out.open("a") as f_pred, skipped_path.open("a") as f_skip:
            for fut in asyncio.as_completed([task(r) for r in records]):
                r, pred, error = await fut
                if pred is not None:
                    f_pred.write(pred.model_dump_json(exclude_none=True) + "\n")
                    f_pred.flush()
                    n_pred += 1
                    tok_in += pred.model_extra.get("input_tokens") or 0
                    tok_out += pred.model_extra.get("output_tokens") or 0
                else:
                    f_skip.write(json.dumps({"id": r.id, "error": error}) + "\n")
                    f_skip.flush()
                    n_skip += 1
                done = n_pred + n_skip
                rate = done / max(time.perf_counter() - start, 1e-9)
                print(f"\r  jev: {done}/{len(records)}  skipped {n_skip}  {rate:.1f} rec/s  "
                      f"tokens in {tok_in:,} out {tok_out:,}", end="", flush=True)
                if n_skip >= args.max_errors:
                    raise SystemExit(f"\n{n_skip} failed records, stopping; see {skipped_path}")
    print(f"\n{n_pred} predictions → {out}" + (f"; {n_skip} skipped → {skipped_path}" if n_skip else ""))
    print(f"tokens: input {tok_in:,}, output {tok_out:,} (as reported by the API)")


def cmd_predict(args) -> None:
    records = [r for path in args.data for r in load_records(path)]
    if len({r.id for r in records}) != len(records):
        raise SystemExit("record ids must be unique across --data files")
    if args.types:
        records = [r for r in records if r.type in args.types.split(",")]
    out, skipped_path = Path(args.out), Path(f"{args.out}.skipped.jsonl")
    if not args.resume and not args.dry_run and out.exists() and out.stat().st_size and not args.overwrite:
        raise SystemExit(f"{out} already has predictions: pass --resume to add to it, or --overwrite to start over")
    done = _ids(out) | _ids(skipped_path) if args.resume else set()
    records = [r for r in records[: args.limit] if r.id not in done]
    if not records:
        raise SystemExit("nothing to predict")

    if args.dry_run:
        r = records[0]
        q = _question(r, args.bool_style)
        print(json.dumps({"model": args.model, "state": r.state,
                          "questions": {"q": q.model_dump() if hasattr(q, "model_dump") else repr(q)}},
                         indent=1, ensure_ascii=False, default=str)[:3000])
        print(f"\ndry run: {len(records):,} calls, ~{sum(map(_est_tokens, records)):,} input tokens; nothing sent")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        print(f"resume: {len(done)} records done, {len(records)} to go", flush=True)
    else:
        for path in (out, skipped_path):
            path.unlink(missing_ok=True)
    asyncio.run(_run(args, records, out, skipped_path))
    if skipped_path.exists() and skipped_path.stat().st_size == 0:
        skipped_path.unlink()


def cmd_repeat(args) -> None:
    """Send the first ``--n`` predicted records again and compare: is the API deterministic?"""
    first = load_predictions(args.preds)
    records = [r for path in args.data for r in load_records(path) if r.id in first][: args.n]

    async def go() -> list[Prediction]:
        sem = asyncio.Semaphore(args.concurrency)
        async with _client(args) as client:
            return await asyncio.gather(*(_predict_one(client, r, sem) for r in records))

    again = asyncio.run(go())
    diffs = []
    for p in again:
        a, b = first[p.id], p
        va = np.array([a.prob]) if a.prob is not None else np.array(a.probs)
        vb = np.array([b.prob]) if b.prob is not None else np.array(b.probs)
        diffs.append(float(np.abs(va - vb).max()))
    flips = sum(d > 0 and (np.argmax(first[p.id].probs) != np.argmax(p.probs) if p.probs else
                           (first[p.id].prob >= 0.5) != (p.prob >= 0.5)) for d, p in zip(diffs, again))
    print(f"{len(diffs)} records sent twice: max |Δp| {max(diffs):.2e}, mean {np.mean(diffs):.2e}, "
          f"{sum(d > 1e-6 for d in diffs)} differ, {flips} change the predicted answer")
    models = Counter(first[p.id].model_extra.get("model") for p in again) + Counter(p.model_extra.get("model") for p in again)
    print(f"models that answered: {dict(models)}")


class JevPredictor:
    """JEV behind the ``Predictor`` interface of ``jevlite.probes``: one call per ``predict``, all its
    questions in that call. Answers are cached in ``cache`` (JSONL) by request, so an interrupted run
    resumes without paying again for what it already sent."""

    def __init__(self, args):
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError:
            raise SystemExit("typesafe-sdk is not installed: uv add typesafe-sdk")
        self.name = f"jev:{args.model}"
        self.model = args.model
        self.client = TypeSafeClient(api_key=_api_key(args), model=args.model, timeout=args.timeout)
        self.cache_path = Path(args.cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, list[list[float]]] = {}
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["probs"]
        self.calls = self.cached = self.tokens_in = self.tokens_out = 0
        self.models: Counter = Counter()

    def _key(self, state: State, questions: Sequence[Question]) -> str:
        body = json.dumps([self.model, serialize_state(state), [q.model_dump(exclude={"id"}) for q in questions]],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        key = self._key(state, questions)
        if key in self.cache:
            self.cached += 1
            return [np.asarray(p) for p in self.cache[key]]
        response = self.client.system_one(state=state, questions={f"q{i}": _jev_question(q) for i, q in enumerate(questions)})
        out = [_probs(q, response.answers[f"q{i}"]) for i, q in enumerate(questions)]
        self.calls += 1
        self.tokens_in += response.usage.input_tokens or 0
        self.tokens_out += response.usage.output_tokens or 0
        self.models[response.model] += 1
        with self.cache_path.open("a") as f:
            f.write(json.dumps({"key": key, "model": response.model, "probs": [p.tolist() for p in out]}) + "\n")
        self.cache[key] = [p.tolist() for p in out]
        if (self.calls + self.cached) % 100 == 0:
            print(f"\r  probes: {self.calls:,} calls, {self.cached:,} from cache, tokens in {self.tokens_in:,}",
                  end="", flush=True)
        return out


def cmd_probes(args) -> None:
    """The black-box probes of ``scripts/probe_blackbox.py`` (``jevlite.probes.run_probes``) for JEV."""
    from jevlite.probes import run_probes

    records = [r for path in args.data for r in load_records(path)]
    predictor = JevPredictor(args)
    result = run_probes(predictor, records, args.n_items, args.n_perms, args.seed, args.shortcuts)
    result["api"] = {"calls": predictor.calls, "from_cache": predictor.cached, "input_tokens": predictor.tokens_in,
                     "output_tokens": predictor.tokens_out, "models": dict(predictor.models)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"\n{json.dumps(result, indent=1)}\n→ {args.out}")


def cmd_models(args) -> None:
    async def go():
        async with _client(args) as client:
            return await client.models.list()

    for m in asyncio.run(go()).models:
        print(f"{m.name}\t{m.release_date}\t{m.description}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="stratified sample of the clean test sets (no API calls)")
    s.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    s.add_argument("--per-cell", type=int, default=700, help="records per (split, primitive)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/jev_api/sample.jsonl")
    s.set_defaults(func=cmd_sample)

    t = sub.add_parser("topup", help="more records per source on top of a sample (no API calls)")
    t.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    t.add_argument("--exclude", nargs="+", default=["data/jev_api/sample.jsonl"], help="samples already drawn")
    t.add_argument("--per-source", type=int, default=300, help="target per (source, primitive) on trained sources")
    t.add_argument("--per-new-source", type=int, default=1000, help="target per (source, primitive) in test_ood_source")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--out", default="data/jev_api/sample_topup.jsonl")
    t.set_defaults(func=cmd_topup)

    for name, func, help_ in (("predict", cmd_predict, "JEV predictions for the records"),
                              ("repeat", cmd_repeat, "determinism check on already predicted records"),
                              ("probes", cmd_probes, "black-box probes, as scripts/probe_blackbox.py"),
                              ("models", cmd_models, "list the models the API key can use")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--key-env", default="TYPESAFE_API_KEY", help="env var (in .env) holding the API key")
        p.add_argument("--concurrency", type=int, default=8, help="requests in flight")
        p.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
        p.set_defaults(func=func)
        if name in ("predict", "repeat", "probes"):
            p.add_argument("--data", nargs="+", required=True)
        if name == "probes":
            p.add_argument("--out", required=True)
            p.add_argument("--cache", default="data/jev_api/probes_cache.jsonl", help="answers already paid for")
            p.add_argument("--n-items", type=int, default=200)
            p.add_argument("--n-perms", type=int, default=8)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--shortcuts", type=int, default=200, metavar="N", help="Bool shortcut probe, N records per source; 0 = off")
        if name == "predict":
            p.add_argument("--out", required=True)
            p.add_argument("--types", default=None, help="comma-separated primitives to keep, e.g. bool")
            p.add_argument("--bool-style", choices=BOOL_STYLES, default="statement",
                           help="how a Bool statement is put to JEV: unchanged, as a question, or with yes/no criteria")
            p.add_argument("--limit", type=int, default=None, help="first N records of --data only (pilot)")
            p.add_argument("--resume", action="store_true", help="keep --out and skip the records already in it")
            p.add_argument("--overwrite", action="store_true", help="delete an existing --out and start over")
            p.add_argument("--max-errors", type=int, default=50, help="stop after this many failed records")
            p.add_argument("--dry-run", action="store_true", help="print the first request and the call count; send nothing")
        if name == "repeat":
            p.add_argument("--preds", required=True)
            p.add_argument("--n", type=int, default=30)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
