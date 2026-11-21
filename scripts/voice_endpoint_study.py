"""Endpoint wait t1 − t0 per endpoint setting on public speech, CPU only (todo Phase 9b, task 1).

    # AMI (conversational; turns end where another speaker takes the floor, ``jevlite.voice.ami``);
    # --download fetches the annotations (23 MB) and the headsets of the meetings (~1 GB for the default four)
    uv run --extra voice python scripts/voice_endpoint_study.py --ami data/audio/ami --download \
        --settings short --n 300 --out data/speed/voice/endpoint_ami

    # once: LibriSpeech test-clean (346 MB, CC BY 4.0)
    mkdir -p data/audio && curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | tar xz -C data/audio
    uv run --extra voice python scripts/voice_endpoint_study.py \\
        --librispeech data/audio/LibriSpeech/test-clean --out data/speed/voice/endpoint

Settings (``jevlite.voice.endpoint``): ``vad-<ms>`` = Silero VAD silence only; ``turn-<ms>[-p<P>][-m<ms>]``
= Silero + Smart Turn v3.2 checked after <ms> of silence, threshold P (0.5), timeout ``-m`` or
``--max-silence-ms``. ``--settings tune`` = the Smart Turn tuning grid (``TUNE_SETTINGS``: threshold
0.3 / 0.5 / 0.7 / 0.9 × timeout 3000 / 1200 / 800 ms, plus vad-500 / vad-800); ``--settings short`` =
``SHORT_SETTINGS`` (the shortlist after the LibriSpeech tuning). Gaps between utterances
must be longer than the longest timeout, otherwise a Smart Turn "incomplete" shows up as missed.

Output in ``--out``: ``endpoint.json`` (settings, summaries, per-utterance rows), ``endpoint.csv`` (one row
per setting × utterance), ``report.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from jevlite.voice import ami
from jevlite.voice.audio import build_stream, load_librispeech
from jevlite.voice.endpoint import (DEFAULT_SETTINGS, SHORT_SETTINGS, SR, TUNE_SETTINGS, SileroVAD, SmartTurn,
                                    parse_setting)
from jevlite.voice.endpoint_study import env_info, match, report_markdown, run_setting, summarize, vad_probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--librispeech", help="extracted split, e.g. data/audio/LibriSpeech/test-clean")
    src.add_argument("--ami", help="AMI root: annotations/ + audio/ (see --download)")
    ap.add_argument("--meetings", default=",".join(ami.DEFAULT_MEETINGS), help="AMI meetings")
    ap.add_argument("--download", action="store_true", help="AMI: fetch missing annotations and headsets")
    ap.add_argument("--min-turn-s", type=float, default=0.5, help="AMI: shortest turn kept")
    ap.add_argument("--target-dbfs", type=float, default=-20.0,
                    help="AMI: speech level per speaker after levelling (AGC-like); 'nan' = raw headset levels")
    ap.add_argument("--n", type=int, default=200, help="utterances / turns")
    ap.add_argument("--settings", default=",".join(DEFAULT_SETTINGS), help="comma list, 'tune' or 'short'")
    ap.add_argument("--max-silence-ms", type=int, default=3000, help="Smart Turn timeout")
    ap.add_argument("--gap", default="3.5,4.5", help="min,max silence between utterances, seconds")
    ap.add_argument("--noise-db", type=float, default=-60.0, help="white noise RMS in dBFS; 'nan' = none")
    ap.add_argument("--max-noise-db", type=float, default=-50.0,
                    help="only recordings with a background ≤ this (dB re peak), so t0 is exact; 'nan' = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad-jit", action="store_true", help="Silero TorchScript instead of ONNX")
    ap.add_argument("--turn-variant", default="cpu", choices=["cpu", "gpu"], help="Smart Turn int8 (cpu) or fp32 (gpu)")
    ap.add_argument("--turn-threads", type=int, default=1)
    ap.add_argument("--out", default="data/speed/voice/endpoint")
    args = ap.parse_args()

    presets = {"tune": TUNE_SETTINGS, "short": SHORT_SETTINGS}
    names = presets.get(args.settings) or [s for s in args.settings.split(",") if s.strip()]
    configs = [parse_setting(s, args.max_silence_ms) for s in names]
    gap = tuple(float(x) for x in args.gap.split(","))
    longest = max(c.max_silence_ms if c.smart_turn else c.silence_ms for c in configs)
    if gap[0] * 1000 <= longest:
        raise SystemExit(f"--gap min {gap[0]} s must exceed the longest timeout / silence, {longest} ms")
    noise = None if args.noise_db != args.noise_db else args.noise_db  # nan → no noise

    max_noise = None if args.max_noise_db != args.max_noise_db else args.max_noise_db
    if args.ami:
        meetings = tuple(m for m in args.meetings.split(",") if m.strip())
        if args.download:
            ami.download(args.ami, meetings)
        st = {}
        target = None if args.target_dbfs != args.target_dbfs else args.target_dbfs
        utts = ami.load_ami(args.ami, args.n, meetings, args.seed, args.min_turn_s, target_dbfs=target, stats=st)
        sh = np.asarray(st["end_shift_ms"])
        source = (f"AMI headset turns ({', '.join(meetings)}; floor change or ≥ 2 s pause; t0 refined from the "
                  f"audio: the annotated end was later by p50 {np.percentile(sh, 50):.0f} / p90 "
                  f"{np.percentile(sh, 90):.0f} ms, unchanged in {100 * np.mean(sh == 0):.0f}% of {len(sh)} turns; "
                  + (f"speakers levelled to {args.target_dbfs} dBFS, gains {min(st['gain_db'].values()):+.0f} … "
                     f"{max(st['gain_db'].values()):+.0f} dB)" if st["gain_db"] else "raw levels)"))
        max_noise = None
        print(source, flush=True)
    else:
        utts = load_librispeech(args.librispeech, args.n, args.seed, max_noise)
        source = f"LibriSpeech {Path(args.librispeech).name}"
    stream, segments = build_stream(utts, gap, noise_db=noise, seed=args.seed, trim=not args.ami)
    speech_min = sum(s.end - s.start for s in segments) / SR / 60
    print(f"{len(segments)} utterances, {speech_min:.1f} min speech, stream {len(stream) / SR / 60:.1f} min", flush=True)

    probs, vad_ms = vad_probs(stream, SileroVAD(onnx=not args.vad_jit))
    print(f"VAD done: {vad_ms:.3f} ms per chunk", flush=True)
    turn = SmartTurn(args.turn_variant, threads=args.turn_threads) if any(c.smart_turn for c in configs) else None

    summaries, rows_all, turn_cache = [], [], {}
    for c in configs:
        endpoints = run_setting(stream, probs, c, turn, turn_cache)
        rows = match(segments, endpoints, len(stream))
        summaries.append(summarize(c, rows, endpoints))
        rows_all += [{"setting": c.name, **r} for r in rows]
        lat = summaries[-1]["latency_ms"]
        print(f"{c.name}: p50 {lat.get('p50', float('nan')):.0f} ms, p95 {lat.get('p95', float('nan')):.0f} ms, "
              f"missed {summaries[-1]['missed_pct']:.1f}%, cut {summaries[-1]['premature_utt_pct']:.1f}%", flush=True)

    result = {
        "stream": {"source": source, "utterances": len(segments),
                   "speech_min": speech_min, "gap_s": list(gap), "noise_db": noise, "seed": args.seed,
                   "max_noise_db": max_noise},
        "vad": {"onnx": not args.vad_jit, "ms_per_chunk": vad_ms},
        "smart_turn": f"v3.2 {turn.variant}, {turn.threads} thread(s)" if turn else None,
        "env": env_info(), "settings": summaries, "rows": rows_all,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "endpoint.json").write_text(json.dumps(result, indent=2))
    with open(out / "endpoint.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0]))
        w.writeheader()
        w.writerows(rows_all)
    (out / "report.md").write_text(report_markdown(result))
    print(f"written to {out}/", flush=True)


if __name__ == "__main__":
    main()
