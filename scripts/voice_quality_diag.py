"""Diagnose a stage-1 voice run locally from files fetched off the volume (todo Phase 9b, stage 1).

    uv run --extra voice python scripts/voice_quality_diag.py \
        --streams data/speed/voice/quality/streams --decode data/speed/voice/quality/decode/smoke \
        --stream librispeech-smoke

For every decoded system on the stream: the WER (``jevlite.voice.quality.word_errors``; locally without
jiwer / whisper-normalizer, so close to, not equal to, the Modal numbers), the reference vs the finals
side by side per utterance (LibriSpeech), and for Nemotron timelines the token bookkeeping: all pieces
in the recording vs all pieces in the replayed finals (lost / duplicated pieces show a bug in the turn
assembly, not in the model).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jevlite.voice import quality as Q
from jevlite.voice.asr import EOB, EOU, pieces_text
from jevlite.voice.endpoint import SR


def finals(events: list[dict]) -> list[dict]:
    return [e for e in events if e["kind"] == "final"]


def per_utterance(meta: dict, events: list[dict]) -> list[tuple[str, str]]:
    """(reference, hypothesis) per utterance: finals assigned to the utterance whose span (to the next
    utterance's start) holds the final's time."""
    segs = meta["segments"]
    out = []
    for i, s in enumerate(segs):
        lo = s["start"] / SR
        hi = segs[i + 1]["start"] / SR if i + 1 < len(segs) else float("inf")
        hyp = " ".join(e["text"] for e in finals(events) if lo < e["audio_s"] <= hi)
        out.append((s["text"], hyp))
    return out


def summarize_diag(path: Path) -> None:
    """Table of a ``diag`` stage (``quality/diag/diag.json``): per streaming variant the WER, the step time
    (ms of GPU per step and per audio second), the text length vs the offline text, and the words the live
    text drops / adds vs the offline decode of the same model (offline is the upper bound)."""
    d = json.loads(path.read_text())
    print(f"gpu: {d.get('gpu')}  versions: {d.get('versions')}")
    offline = next((v for v in d["variants"] if "offline_text" in v), None)
    if offline:
        print(f"nemotron offline per utterance: WER {offline['offline_wer']:.1f}%")
    print(f"\n{'model':9} {'step ms':>7} {'graphs':>6} {'WER %':>6} {'GPU ms/step':>11} {'GPU ms/audio s':>14} "
          f"{'words':>5} {'wall s':>6}")
    for v in d["variants"]:
        if "error" in v or "wer" not in v:
            print(f"{v.get('model'):9} {1000 * (v.get('step_s') or 0):7.0f} ERROR {v.get('error', '')[:80]}")
            continue
        step = v["step_s"] or 0.0
        per_s = v["step_ms_p50"] / step if step else float("nan")
        print(f"{v['model']:9} {1000 * step:7.0f} {str(v['cuda_graphs']):>6} {v['wer']:6.1f} {v['step_ms_p50']:11.1f} "
              f"{per_s:14.0f} {len(v['text'].split()):5d} {v['wall_s']:6.1f}")
    if offline:
        ref = Counter(Q.normalizer()(offline["offline_text"]).split())
        print(f"\nvs nemotron offline ({sum(ref.values())} words):")
        for v in d["variants"]:
            if v.get("model") != "nemotron" or "text" not in v:
                continue
            hyp = Counter(Q.normalizer()(v["text"]).split())
            print(f"  step {1000 * v['step_s']:.0f} ms graphs={v['cuda_graphs']}: missing {sum((ref - hyp).values())}, "
                  f"extra {sum((hyp - ref).values())}")


L4_USD_PER_H = 0.80  # Modal's L4 list price (approximate; CPU / memory not included)


def summarize_quality(path: Path) -> None:
    """A fetched ``quality/<run>/quality.json``: GPU time per decode job and its approximate price, the batch
    checks, and per system × source: the final's wait after the endpoint (t2 − t1, p50), the AMI calls per
    turn and the share of turns whose end the system never marks."""
    q = json.loads(path.read_text())
    dec = q.get("decode", {})
    total = 0.0
    print(f"gpu: {q.get('gpu')}")
    for job, v in dec.get("jobs", {}).items():
        total += v["wall_s"]
        extra = (f", batch {v['batch']}, {v['steps']} steps, {v['step_ms_p50']:.1f} ms/step p50, "
                 f"non-prefix steps {v['non_prefix_steps']}" if "batch" in v else "")
        print(f"  {job:18} {v['wall_s'] / 60:5.1f} min{extra}")
    for job, c in dec.get("batch_check", {}).items():
        total += c["single_s"] + c["batched_s"]
        print(f"  check {job:12} {(c['single_s'] + c['batched_s']) / 60:5.1f} min, ok={c['ok']}, "
              f"disagree {c['disagree_pct']}")
    print(f"decode work {total / 60:.1f} min ≈ ${total / 3600 * L4_USD_PER_H:.2f} at ${L4_USD_PER_H}/h "
          "(+ container start and model loading, not timed)")
    print(f"\n{'source':11} {'system':22} {'t1 p50':>6} {'t2 p50':>6} {'t2-t1':>5} {'t1 p95':>6} "
          f"{'calls/turn':>10} {'missed %':>8} {'WER %':>6} {'GPU ms/s':>8}")
    for r in q["summaries"]:
        t1, t2 = r["latency_ms"], r["t2_ms"]
        calls = r.get("calls_per_turn")
        missed = r.get("missed_pct", 0.0)
        print(f"{r['source']:11} {r['setting']:22} {t1['p50']:6.0f} {t2['p50']:6.0f} {t2['p50'] - t1['p50']:5.0f} "
              f"{t1['p95']:6.0f} {(f'{calls:.2f}' if calls is not None else '-'):>10} {missed:8.1f} "
              f"{r['wer']:6.1f} {r['compute_ms_per_audio_s']:8.1f}")


T3_MAX_MS = 1000.0  # the run's "good" bar: ASR keeps up and t3 − t1 p95 within this


def _good(r: dict) -> bool:
    return bool(r["asr_keeps_up"]) and r["t3_minus_t1_ms"].get("p95", 0.0) <= T3_MAX_MS


def summarize_capacity(paths: list[Path]) -> None:
    """Fetched ``capacity/<gpu>-<stamp>/capacity.json`` files: per GPU × JEV input length the streams one GPU
    carries (last N that kept up with t3 − t1 p95 ≤ 1 s, and the first N that did not), the latency there, and
    where the GPU time goes at that N: ASR = batched step p50 / step length (includes waiting on JEV kernels),
    JEV = calls/s × forward ms / calls per forward. Also the JEV-only bound: calls/s JEV alone takes at p95 ≤
    300 ms over the measured calls per stream per second (what the GPU would carry if ASR ran elsewhere)."""
    stage1_t1 = {"p50": 424.0, "p95": 749.0}  # vad-300, stage 1
    print(f"{'gpu':22} {'L':>5} {'N ok':>5} {'N fail':>6} {'calls/s':>7} {'t2-t1 p50/p95':>13} {'t3-t1 p50/p95':>13} "
          f"{'t3-t0 p50/p95':>13} {'ASR %':>5} {'JEV %':>5} {'JEV-only N':>10}")
    for path in paths:
        d = json.loads(path.read_text())
        gpu = d["gpu"].replace("NVIDIA ", "")
        jev300 = {}
        for j in d["jev"]:
            ok = [x for x in j["load"] if x["latency_ms"].get("p95", float("inf")) <= 300 and not x.get("incomplete")]
            jev300[j["length"]] = max((x["achieved_per_s"] for x in ok), default=0.0)
        for length in sorted({r["length"] for r in d["live"]}):
            rows = sorted((r for r in d["live"] if r["length"] == length), key=lambda r: r["streams"])
            good = [r for r in rows if _good(r)]
            bad = [r for r in rows if not _good(r)]
            fail_n = min((r["streams"] for r in bad), default=None)
            if not good:
                print(f"{gpu:22} {length:5d} {'<' + str(fail_n):>5} {fail_n:6d}")
                continue
            r = max(good, key=lambda r: r["streams"])
            step = 1000 * r["step_s"]
            asr = 100 * r["step_ms"]["p50"] / step
            jev = 100 * r["jev_calls_per_s"] * r["jev_forward_ms"]["mean"] / r["jev_batch_requests"]["mean"] / 1000
            per_stream = r["jev_calls_per_s"] / r["streams"]
            t2, t3 = r["t2_minus_t1_ms"], r["t3_minus_t1_ms"]
            print(f"{gpu:22} {length:5d} {r['streams']:5d} {str(fail_n or '-'):>6} {r['jev_calls_per_s']:7.1f} "
                  f"{t2['p50']:6.0f}/{t2['p95']:<6.0f} {t3['p50']:6.0f}/{t3['p95']:<6.0f} "
                  f"{stage1_t1['p50'] + t3['p50']:6.0f}/{stage1_t1['p95'] + t3['p95']:<6.0f} {asr:5.0f} {jev:5.0f} "
                  f"{jev300.get(length, 0.0) / per_stream:10.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", help="summarize a fetched quality/diag/diag.json and exit")
    ap.add_argument("--quality", help="summarize a fetched quality/<run>/quality.json and exit")
    ap.add_argument("--capacity", nargs="+", help="summarize fetched capacity/<gpu>-<stamp>/capacity.json files and exit")
    ap.add_argument("--streams")
    ap.add_argument("--decode")
    ap.add_argument("--stream", default="librispeech-smoke")
    ap.add_argument("--setting", default="vad-500", help="detector for the Nemotron replay")
    ap.add_argument("--show", type=int, default=10, help="utterances to print per system")
    args = ap.parse_args()
    if args.diag:
        summarize_diag(Path(args.diag))
        return
    if args.quality:
        summarize_quality(Path(args.quality))
        return
    if args.capacity:
        summarize_capacity([Path(p) for p in args.capacity])
        return
    if not (args.streams and args.decode):
        ap.error("--streams and --decode are required without --diag")

    streams, decode = Path(args.streams), Path(args.decode)
    meta = json.loads((streams / f"{args.stream}.json").read_text())
    import numpy as np

    probs = np.load(streams / f"{args.stream}.vad.npy")
    for d in sorted(p for p in decode.iterdir() if p.is_dir()):
        f = d / f"{args.stream}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        print(f"\n=== {d.name}")
        if d.name.startswith("nemotron-r"):
            r = int(d.name.split("-r")[1])
            recorded = [p for ps, _ in rec["steps"] for p in ps if p not in (EOU, EOB)]
            out = Q.replay_nemotron(rec, probs, args.setting, r, int(round(rec["audio_s"] * SR)))
            events = out["events"]
            replay_text = " ".join(e["text"] for e in finals(events))
            print(f"replayed under {args.setting}: {len(finals(events))} finals")
            print(f"recorded pieces: {len(recorded)}; recorded text words: {len(pieces_text(recorded).split())}; "
                  f"finals words: {len(replay_text.split())}")
            rec_words, fin_words = Counter(pieces_text(recorded).split()), Counter(replay_text.split())
            lost, extra = rec_words - fin_words, fin_words - rec_words
            print(f"words in the recording but not in the finals: {sum(lost.values())} {dict(list(lost.items())[:20])}")
            print(f"words in the finals but not in the recording: {sum(extra.values())} {dict(list(extra.items())[:20])}")
            err, n = Q.word_errors(meta["reference"], pieces_text(recorded))
            print(f"WER of the whole recorded text (no turns): {100 * err / max(n, 1):.1f}% ({err}/{n})")
        else:
            events = rec["events"]
        err, n = Q.word_errors(meta["reference"], " ".join(e["text"] for e in finals(events)))
        print(f"WER of the finals: {100 * err / max(n, 1):.1f}% ({err}/{n})")
        for ref, hyp in per_utterance(meta, events)[:args.show]:
            e, m = Q.word_errors(ref, hyp)
            if e:
                print(f"  [{e}/{m}] REF: {ref}\n         HYP: {hyp}")


if __name__ == "__main__":
    main()
