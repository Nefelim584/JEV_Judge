"""Endpoint cost table on AMI pauses: latency vs extra JEV calls per setting, CPU only (todo Phase 9b,
"Pause labels and the cost table"; ``jevlite.voice.pauses`` has the definitions).

    # 1) labels only (no model, seconds) + ~30 clips for the listening check
    uv run --extra voice python scripts/voice_pause_study.py --ami data/audio/ami --labels-only \
        --out data/speed/voice/pauses

    # 2) the table: every setting over every speaker's whole headset stream (~10–15 min on a laptop CPU)
    uv run --extra voice python scripts/voice_pause_study.py --ami data/audio/ami --out data/speed/voice/pauses

The AMI data comes from ``scripts/voice_endpoint_study.py --ami … --download`` (same layout; ``--download``
works here too). Output in ``--out``: ``report.md``, ``pauses.json`` (settings, summaries, label stats),
``pauses.csv`` (every labelled pause), ``endpoints.csv`` (setting × endpoint, matched), and ``check/``
(the listening clips + ``check/pauses.csv`` with a verdict column to fill in).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jevlite.voice import ami
from jevlite.voice import pauses as P
from jevlite.voice.endpoint import SHORT_SETTINGS, SileroVAD, SmartTurn, parse_setting
from jevlite.voice.endpoint_study import env_info, run_setting, vad_probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ami", required=True, help="AMI root: annotations/ + audio/")
    ap.add_argument("--meetings", default=",".join(ami.DEFAULT_MEETINGS))
    ap.add_argument("--download", action="store_true", help="fetch missing annotations and headsets")
    ap.add_argument("--settings", default="pause", help="comma list, 'pause' (PAUSE_SETTINGS) or 'short'")
    ap.add_argument("--max-silence-ms", type=int, default=3000, help="Smart Turn timeout when the name has no -m")
    ap.add_argument("--min-pause-ms", type=int, default=round(P.MIN_PAUSE_S * 1000))
    ap.add_argument("--min-turn-s", type=float, default=0.5, help="'long' turns for the second missed column")
    ap.add_argument("--target-dbfs", type=float, default=-20.0, help="speaker levelling; 'nan' = raw levels")
    ap.add_argument("--no-mute", action="store_true", help="keep the other speakers' crosstalk in each stream")
    ap.add_argument("--noise-db", type=float, default=-60.0, help="white noise RMS in dBFS; 'nan' = none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=30, help="clips for the listening check (0 = none)")
    ap.add_argument("--labels-only", action="store_true", help="labels, stats and clips only: no VAD, no Smart Turn")
    ap.add_argument("--vad-jit", action="store_true", help="Silero TorchScript instead of ONNX")
    ap.add_argument("--turn-variant", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--turn-threads", type=int, default=1)
    ap.add_argument("--out", default="data/speed/voice/pauses")
    args = ap.parse_args()

    meetings = tuple(m for m in args.meetings.split(",") if m.strip())
    if args.download:
        ami.download(args.ami, meetings)
    target = None if args.target_dbfs != args.target_dbfs else args.target_dbfs
    noise = None if args.noise_db != args.noise_db else args.noise_db
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mts = []
    for m in meetings:
        mt = P.load_meeting(args.ami, m, min_pause_s=args.min_pause_ms / 1000, target_dbfs=target)
        mts.append(mt)
        print(f"{m}: {mt.duration_s / 60:.1f} min, {len(mt.channels)} speakers, pauses "
              f"{P.pause_stats(mt.pauses)['counts']}", flush=True)
    all_pauses = [p for mt in mts for p in mt.pauses]
    stats = P.pause_stats(all_pauses)
    with open(out / "pauses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*asdict(all_pauses[0]), "ms"])
        w.writeheader()
        w.writerows({**asdict(p), "ms": round(p.ms)} for p in all_pauses)
    if args.check:
        P.export_check(args.ami, mts, out / "check", n=args.check, seed=args.seed)
        print(f"listening check: {args.check} clips in {out / 'check'}/ (fill the verdict column of pauses.csv)")
    description = (f"AMI {', '.join(meetings)}, every speaker's whole headset stream ("
                   f"{sum(mt.duration_s * len(mt.channels) for mt in mts) / 60:.0f} min of audio), "
                   + (f"speakers levelled to {target} dBFS, " if target is not None else "raw levels, ")
                   + ("other speakers muted, " if not args.no_mute else "crosstalk kept, ")
                   + (f"noise {noise} dBFS" if noise is not None else "no noise"))
    source = {"description": description, "meetings": list(meetings), "min_pause_ms": args.min_pause_ms,
              "min_turn_s": args.min_turn_s, "mute_others": not args.no_mute, "target_dbfs": target,
              "noise_db": noise, "seed": args.seed}
    print(f"all: {stats['counts']}; ideal silence detector, extra calls per turn: "
          + ", ".join(f"{t} ms {v:.2f}" for t, v in stats["ideal_vad_extra_per_turn"].items()), flush=True)
    if args.labels_only:
        (out / "pauses.json").write_text(json.dumps({"source": source, "pause_stats": stats}, indent=2))
        print(f"written to {out}/ (labels only)")
        return

    presets = {"pause": P.PAUSE_SETTINGS, "short": SHORT_SETTINGS}
    names = presets.get(args.settings) or [s for s in args.settings.split(",") if s.strip()]
    configs = [parse_setting(s, args.max_silence_ms) for s in names]
    vad = SileroVAD(onnx=not args.vad_jit)
    turn = SmartTurn(args.turn_variant, threads=args.turn_threads) if any(c.smart_turn for c in configs) else None
    ep_all = {c.name: [] for c in configs}
    pause_all = {c.name: [] for c in configs}
    call_ms = {c.name: [] for c in configs}
    vad_ms = []
    for k, mt in enumerate(mts):
        for spk in sorted(mt.channels):
            stream = P.speaker_stream(args.ami, mt, spk, mute_others=not args.no_mute, noise_db=noise,
                                      seed=args.seed + k)
            vad.reset()
            probs, ms = vad_probs(stream, vad)
            vad_ms.append(ms)
            sp = [p for p in mt.pauses if p.speaker == spk]
            cache: dict = {}
            for c in configs:
                eps = run_setting(stream, probs, c, turn, cache)
                er, pr = P.match_pauses(sp, mt.active[spk], eps)
                ep_all[c.name] += [{"meeting": mt.name, "speaker": spk, **r} for r in er]
                pause_all[c.name] += pr
                call_ms[c.name] += [e.turn_ms / e.turn_calls for e in eps if e.turn_calls]
            print(f"{mt.name}.{spk}: {len(stream) / 16000 / 60:.1f} min, {len(sp)} pauses", flush=True)

    summaries = []
    for c in configs:
        s = P.summarize_setting(c.name, ep_all[c.name], pause_all[c.name], args.min_turn_s)
        s["config"] = asdict(c)
        if call_ms[c.name]:
            s["turn_call_ms"] = {"p50": float(np.percentile(call_ms[c.name], 50)),
                                 "p95": float(np.percentile(call_ms[c.name], 95))}
        summaries.append(s)
        lat = s["latency_ms"]
        print(f"{c.name}: p50 {lat['p50'] or float('nan'):.0f} / p95 {lat['p95'] or float('nan'):.0f} ms, "
              f"JEV calls per turn {s['calls_per_turn']:.2f} (extra {s['extra_per_turn']:.2f}), "
              f"missed {s['missed_pct']:.1f}%", flush=True)

    result = {"source": source, "vad": {"onnx": not args.vad_jit, "ms_per_chunk": float(np.mean(vad_ms))},
              "smart_turn": f"v3.2 {turn.variant}, {turn.threads} thread(s)" if turn else None,
              "env": env_info(), "pause_stats": stats, "settings": summaries}
    (out / "pauses.json").write_text(json.dumps(result, indent=2))
    rows = [{"setting": n, **r} for n, rs in ep_all.items() for r in rs]
    if rows:
        with open(out / "endpoints.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "report.md").write_text(P.report_markdown(result))
    print(f"written to {out}/", flush=True)


if __name__ == "__main__":
    main()
