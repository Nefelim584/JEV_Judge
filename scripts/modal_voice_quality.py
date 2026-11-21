"""Voice stage 1 on Modal: quality of ASR + end of turn on one GPU (todo Phase 9b; the library is
``jevlite.voice.quality``). Runs without the laptop: the app is deployed once, a stage is started with
``spawn`` (the launch command returns at once; the call lives on Modal), results go to the volume
``jevlite-voice``. (``modal run --detach`` was not enough: on 2026-11-20 a client disconnect cancelled
the orchestrator and, with it, the GPU call.)

    # once, and again after every code change (a deployed app with no calls runs no containers: free):
    uv run --extra bench modal deploy scripts/modal_voice_quality.py

    # start a stage (returns at once):
    uv run --extra bench python scripts/modal_voice_quality.py diag      # Nemotron WER diagnosis, ~6 min L4
    uv run --extra bench python scripts/modal_voice_quality.py smoke     # 10 utterances, every system
    uv run --extra bench python scripts/modal_voice_quality.py full      # the whole study, ~30 min L4

    # follow / stop / fetch:
    uv run --extra bench modal app logs jevlite-voice-quality
    uv run --extra bench modal app stop jevlite-voice-quality            # stops the running calls too
    uv run --extra bench modal volume get jevlite-voice quality data/speed/voice

Stages: ``smoke`` (data + models into the volume on the CPU, then ~10 LibriSpeech utterances through every
system on an L4, the NeMo checks and the projected GPU time of the full run); ``diag`` (the smoke stream
through Nemotron / Parakeet variants: step 80 vs 160 ms, CUDA-graph decoder on / off, determinism, offline
upper bound); ``full`` (AMI test meetings, 146 min of shortened headset audio in 16 streams, + 100 LibriSpeech
utterances, 14 min). The NeMo models step at their own chunk (160 ms; the diag of 2026-11-20: Nemotron
at 80 ms lost most words) and decode all streams of a job in one batch (the longest stream, 23 min, sets
the time), after a check on the smoke stream that the batch decodes as a single stream does. ``modal run scripts/modal_voice_quality.py --stage …`` still works, tied to the terminal.

Functions: ``prepare`` (CPU: AMI annotations + 16 headsets, LibriSpeech test-clean, the three models, the
streams with their Silero P(speech)); ``decode`` (GPU, ``--gpu``, default L4: the models only; existing results are skipped,
so a re-run resumes — per stream for Whisper, per job for the batched NeMo jobs);
``evaluate`` (CPU: Nemotron replays under ``vad-300/500/800``, metrics, ``report.md``); ``orchestrate`` (a
0.25-CPU container that calls them in order, so a closed laptop does not stop the run).

Image: CUDA 12.8 + cuDNN 9 runtime (faster-whisper's CTranslate2 needs both), torch 2.8.0, NeMo 3.0.0 [asr]
(its own streaming pipeline was checked against ``jevlite.voice.asr`` on 2026-11-20), faster-whisper 1.2.1,
silero-vad 6.2.3, the ``jevlite`` source.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "jevlite-voice-quality"
ROOT = Path("/vol")
DATA, STREAMS, DECODE, QUALITY = ROOT / "data", ROOT / "streams", ROOT / "decode", ROOT / "quality"
SYSTEMS = ("parakeet-eou", "nemotron+vad-300", "nemotron+vad-500", "nemotron+vad-800", "whisper+vad-500")
MEETINGS = ("ES2004a", "IS1009a", "TS3003a", "EN2002a")
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SMOKE_ID = "librispeech-smoke"

app = modal.App(APP)
vol = modal.Volume.from_name("jevlite-voice", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("libsndfile1", "ffmpeg", "git", "build-essential", "curl")
    .pip_install("Cython", "packaging")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", "nemo_toolkit[asr]==3.0.0", "faster-whisper==1.2.1",
                 "silero-vad==6.2.3", "onnxruntime>=1.20", "soundfile>=0.12", "jiwer>=3.1", "huggingface_hub")
    .env({"HF_HOME": str(ROOT / "hf"), "TOKENIZERS_PARALLELISM": "false", "NEMO_CACHE_DIR": str(ROOT / "nemo")})
    .add_local_python_source("jevlite")
)


def _log(s: str) -> None:
    print(s, flush=True)


@app.function(image=image, cpu=4.0, memory=8192, timeout=3600, volumes={str(ROOT): vol})
def prepare(meetings: list[str], n_librispeech: int, seed: int = 0) -> list[dict]:
    """Data, models and streams into the volume; skipped if already done with the same settings."""
    import subprocess

    from jevlite.voice import ami, quality
    from jevlite.voice.asr import NEMOTRON, PARAKEET_EOU, WHISPER
    from jevlite.voice.endpoint import SileroVAD

    vol.reload()
    cfg = {"meetings": meetings, "n_librispeech": n_librispeech, "seed": seed, "version": 1}
    marker = STREAMS / "prepare.json"
    if marker.exists() and json.loads(marker.read_text()).get("config") == cfg:
        _log("streams already prepared")
        return json.loads((STREAMS / "index.json").read_text())

    # AMI: uploaded once from the laptop (the Edinburgh mirror answered 503 for the audio, 2026-11-18 and -20):
    #   modal volume put jevlite-voice <dir with annotations/ and audio/> ami_upload; else downloaded here
    ami_root = ROOT / "ami_upload" if (ROOT / "ami_upload" / "annotations").exists() else DATA / "ami"
    ami.download(ami_root, meetings, log=_log)
    ls = DATA / "LibriSpeech" / "test-clean"
    if not ls.exists():
        _log("downloading LibriSpeech test-clean (346 MB)")
        DATA.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"curl -sL --retry 5 {LIBRISPEECH_URL} | tar xz -C {DATA}", shell=True, check=True)
    vol.commit()

    from faster_whisper.utils import download_model
    from huggingface_hub import snapshot_download

    for repo in (PARAKEET_EOU, NEMOTRON):
        _log(f"model {repo}")
        snapshot_download(repo, allow_patterns=["*.nemo", "*.json", "*.md"])
    _log(f"model faster-whisper {WHISPER}")
    download_model(WHISPER)
    vol.commit()

    vad = SileroVAD(onnx=True)
    index = quality.prepare_librispeech(ls, STREAMS, n=10, vad=vad, seed=seed + 1, sid=SMOKE_ID)
    index += quality.prepare_librispeech(ls, STREAMS, n=n_librispeech, vad=vad, seed=seed)
    index += quality.prepare_ami(ami_root, STREAMS, meetings=meetings, vad=vad, seed=seed, log=_log)
    (STREAMS / "index.json").write_text(json.dumps(index, indent=1))
    marker.write_text(json.dumps({"config": cfg, "minutes": {s["id"]: s["duration_s"] / 60 for s in index}}, indent=1))
    vol.commit()
    _log(f"{len(index)} streams, {sum(s['duration_s'] for s in index) / 60:.0f} min")
    return index


def _smoke_checks(pool, audio, segments) -> dict:
    """NeMo checks on the first two utterances: live loop vs the model's offline transcript, the hypothesis
    semantics, markers in the vocabulary, the streaming geometry; Whisper on the same audio."""
    import tempfile
    import time

    import numpy as np
    import soundfile as sf

    from jevlite.voice.asr import EOB, EOU, NEMOTRON, PARAKEET_EOU, NemoStream, pieces_text, record_steps
    from jevlite.voice.endpoint import SR

    a, b = segments[0]["start"], segments[1]["end"]
    clip = audio[a:b]
    out = {"reference": f"{segments[0]['text']} {segments[1]['text']}", "clip_s": len(clip) / SR}
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, clip, SR)
        for name in (PARAKEET_EOU, NEMOTRON):
            r = {}
            try:
                spec = pool.nemo(name, 1)
                sc = spec.model.encoder.streaming_cfg
                r["streaming_cfg"] = {k: str(getattr(sc, k, None)) for k in
                                      ("chunk_size", "shift_size", "pre_encode_cache_size", "drop_extra_pre_encoded")}
                r["buffer_s"], r["lookahead_steps"] = spec.buffer_s, spec.lookahead_steps
                vocab = list(getattr(spec.model.tokenizer, "vocab", []) or [])
                r["eou_in_vocab"], r["eob_in_vocab"] = EOU in vocab, EOB in vocab
                s = NemoStream(spec)
                t = time.perf_counter()
                steps = record_steps(s, clip, flush_steps=spec.lookahead_steps)
                r["live_s"] = time.perf_counter() - t
                pieces = [p for ps, _ in steps for p in ps]
                r["live_text"] = pieces_text(pieces)
                r["live_markers"] = [p for p in pieces if p in (EOU, EOB)]
                ms = np.array([m for _, m in steps])
                r["step_ms"] = {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
                                "first": float(ms[0])}
                r["non_prefix_steps"] = s.non_prefix_steps
                t = time.perf_counter()
                off = spec.model.transcribe([f.name], batch_size=1)
                off = off[0] if isinstance(off, (list, tuple)) else off
                off = off[0] if isinstance(off, (list, tuple)) else off
                r["offline_text"] = getattr(off, "text", str(off))
                r["offline_s"] = time.perf_counter() - t
            except Exception as e:  # keep going: the smoke must report everything it can
                import traceback

                r["error"] = f"{type(e).__name__}: {e}"
                r["traceback"] = traceback.format_exc()[-3000:]
            out[name] = r
            _log(f"{name}: {json.dumps({k: v for k, v in r.items() if k != 'traceback'})[:1500]}")
    try:
        t = time.perf_counter()
        w = pool.whisper_turns()
        out["whisper_load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        out["whisper_text"] = w.finalize(clip, 0)
        out["whisper_s"] = time.perf_counter() - t
    except Exception as e:
        out["whisper_error"] = f"{type(e).__name__}: {e}"
    _log(f"whisper: {out.get('whisper_text', out.get('whisper_error'))}")
    return out


BATCH_CHECK_MAX_DIFF = 1.0  # % words: batched vs single-stream text on the smoke stream, else the run stops


def _batch_check(spec, kind: str, audio, reference: str) -> dict:
    """Before a batched job: the smoke stream and its first half each decoded alone (``NemoStream``, the
    path the diag validated) vs ``[whole, half, half]`` on two slots — the second half goes to the slot the
    first half freed, so it tests the slot reset. B1: the whole decoded text; A1: the finals. ``disagree_pct``
    = word errors of each batched text against the same audio's single-stream text; ``ok`` when every one is
    ≤ ``BATCH_CHECK_MAX_DIFF``. Also the WER vs the reference, single and batched, of the whole stream."""
    import time

    from jevlite.voice import quality
    from jevlite.voice.asr import EOB, EOU, NemoStream, ParakeetEOU, pieces_text, record_steps

    def text_of(res) -> str:
        if kind == "nemotron":
            return pieces_text([p for ps, _ in res["steps"] for p in ps if p not in (EOU, EOB)])
        return " ".join(e["text"] for e in res["events"] if e["kind"] == "final")

    def single(x):
        if kind == "nemotron":
            return {"steps": record_steps(NemoStream(spec), x, flush_steps=spec.lookahead_steps)}
        return quality.run_live(ParakeetEOU(NemoStream(spec)), x)

    def pct(ref: str, hyp: str) -> float:
        e, n = quality.word_errors(ref, hyp)
        return 100 * e / max(n, 1)

    half = audio[:len(audio) // 2]
    t = time.perf_counter()
    whole_text, half_text = text_of(single(audio)), text_of(single(half))
    single_s = time.perf_counter() - t
    t = time.perf_counter()
    res, binfo = quality.decode_batched(spec, [audio, half, half], kind, batch=2)
    batched_s = time.perf_counter() - t
    texts = [text_of(r) for r in res]
    disagree = [pct(whole_text, texts[0]), pct(half_text, texts[1]), pct(half_text, texts[2])]
    return {"single_wer": pct(reference, whole_text), "batched_wer": pct(reference, texts[0]),
            "disagree_pct": disagree, "half_words": len(half_text.split()), "single_s": single_s,
            "batched_s": batched_s, "batch_info": binfo, "ok": all(d <= BATCH_CHECK_MAX_DIFF for d in disagree)}


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, timeout=2 * 3600, volumes={str(ROOT): vol})
def decode(jobs: list[str], stream_ids: list[str], run: str, smoke: bool = False) -> dict:
    """The model work. Job ``parakeet-eou`` / ``whisper+<setting>``: events; ``nemotron-r<R>``: the token
    timeline (replayed by ``evaluate``). One JSON per job × stream in ``decode/<run>/<job>/``. The NeMo jobs
    decode all their streams at once (``quality.decode_batched``, one slot per stream, the step = the model
    chunk, 160 ms), after ``_batch_check`` passes on the smoke stream; Whisper decodes stream by stream."""
    import time

    import torch

    from jevlite.voice import quality
    from jevlite.voice.asr import NEMOTRON, PARAKEET_EOU, ModelPool

    vol.reload()
    pool = ModelPool(device="cuda")
    info = {"gpu": torch.cuda.get_device_name(0), "versions": _versions(), "jobs": {}, "batch_check": {}}
    smoke_audio, smoke_meta, _ = quality.load_stream(STREAMS, SMOKE_ID)
    if smoke:
        info["checks"] = _smoke_checks(pool, smoke_audio, smoke_meta["segments"])
    for job in jobs:
        d = DECODE / run / job
        d.mkdir(parents=True, exist_ok=True)
        todo = [sid for sid in stream_ids if not (d / f"{sid}.json").exists()]
        if not todo:
            continue
        t_job = time.perf_counter()
        if job == "parakeet-eou" or job.startswith("nemotron-r"):
            kind = "parakeet" if job == "parakeet-eou" else "nemotron"
            spec = pool.nemo(PARAKEET_EOU, 1) if kind == "parakeet" else pool.nemo(NEMOTRON, int(job.split("-r")[1]))
            check = _batch_check(spec, kind, smoke_audio, smoke_meta["reference"])
            info["batch_check"][job] = check
            _log(f"{job} batch check: {json.dumps(check)}")
            if not check["ok"]:
                (DECODE / run / "decode_info.json").write_text(json.dumps(info, indent=1))
                vol.commit()
                raise RuntimeError(f"{job}: batched decoding differs from the single stream, stopping: {check}")
            audios = [quality.load_stream(STREAMS, sid)[0] for sid in todo]
            _log(f"{job}: {len(todo)} streams, {sum(len(a) for a in audios) / 16000 / 60:.0f} min, batched")
            results, binfo = quality.decode_batched(spec, audios, kind, log=_log)
            for sid, res in zip(todo, results):
                res["wall_s"] = binfo["wall_s"]
                (d / f"{sid}.json").write_text(json.dumps(res))
            vol.commit()
            info["jobs"][job] = binfo
            _log(f"{job}: {json.dumps(binfo)}")
            continue
        if not job.startswith("whisper+"):
            raise ValueError(job)
        for sid in todo:
            audio, meta, probs = quality.load_stream(STREAMS, sid)
            t = time.perf_counter()
            res = quality.run_whisper(pool.whisper_turns().model, audio, probs, job.split("+", 1)[1])
            res["wall_s"] = time.perf_counter() - t
            (d / f"{sid}.json").write_text(json.dumps(res))
            vol.commit()
            _log(f"{job} {sid}: {res['audio_s'] / 60:.1f} min audio, compute {res['compute_ms'] / 1000:.0f} s, "
                 f"wall {res['wall_s']:.0f} s")
        info["jobs"][job] = {"wall_s": time.perf_counter() - t_job}
    (DECODE / run / "decode_info.json").write_text(json.dumps(info, indent=1))
    vol.commit()
    return info


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ("nemo_toolkit", "torch", "faster-whisper", "ctranslate2", "silero-vad", "onnxruntime"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


@app.function(image=image, cpu=2.0, memory=8192, timeout=3600, volumes={str(ROOT): vol})
def evaluate(run: str, stream_ids: list[str], full_minutes: float | None = None) -> str:
    """Metrics and the report from ``decode/<run>`` into ``quality/<run>/``."""
    from jevlite.voice import quality

    vol.reload()
    info = json.loads((DECODE / run / "decode_info.json").read_text())
    rows = quality.evaluate(STREAMS, DECODE / run, stream_ids=stream_ids, log=_log)
    result = {"gpu": info.get("gpu"), "versions": info.get("versions"), "summaries": rows, "decode": info}
    md = quality.report_markdown(result)
    if info.get("checks"):
        md += "\n## Smoke checks\n\n```json\n" + json.dumps(
            {k: ({kk: vv for kk, vv in v.items() if kk != "traceback"} if isinstance(v, dict) else v)
             for k, v in info["checks"].items()}, indent=1)[:12000] + "\n```\n"
    if full_minutes:  # projected GPU time of the full run from this one's compute per audio second
        lines = ["", "## Projection for the full run", "", f"Full audio: {full_minutes:.0f} min.", ""]
        for r in rows:
            if r["source"] == "librispeech":
                lines.append(f"- {r['setting']}: {r['compute_ms_per_audio_s']:.0f} ms per audio s → "
                             f"~{r['compute_ms_per_audio_s'] * full_minutes * 60 / 1000 / 60:.0f} GPU min")
        md += "\n".join(lines) + "\n"
    out = QUALITY / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(json.dumps(result, indent=1))
    (out / "report.md").write_text(md)
    vol.commit()
    return md


DIAG_VARIANTS = (  # (model, step_s or None = the model's own chunk, CUDA-graph decoder)
    ("nemotron", 0.08, True), ("nemotron", 0.08, True), ("nemotron", 0.08, False), ("nemotron", None, True),
    ("parakeet", 0.08, True), ("parakeet", None, True),
)


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, timeout=1800, volumes={str(ROOT): vol})
def diag() -> dict:
    """Why does Nemotron lose words in the stream (smoke: WER 13% vs 0.6% Parakeet, the turn assembly ruled
    out)? The smoke stream through each variant of ``DIAG_VARIANTS`` (the second run of the first variant
    tests determinism); Nemotron: the whole recorded text vs the reference; Parakeet: the finals (it needs
    its reset after <EOU>); plus Nemotron offline on each utterance as the upper bound."""
    import tempfile
    import time

    import numpy as np
    import soundfile as sf

    from jevlite.voice import quality
    from jevlite.voice.asr import (EOB, EOU, NEMOTRON, PARAKEET_EOU, NemoStream, ParakeetEOU, load_nemo,
                                   pieces_text, record_steps)
    from jevlite.voice.endpoint import SR

    vol.reload()
    audio, meta, _ = quality.load_stream(STREAMS, SMOKE_ID)
    ref = meta["reference"]
    out = {"gpu": None, "versions": _versions(), "variants": []}
    import torch

    out["gpu"] = torch.cuda.get_device_name(0)
    for model, step_s, graphs in DIAG_VARIANTS:
        r = {"model": model, "step_s": step_s, "cuda_graphs": graphs}
        try:
            spec = load_nemo(NEMOTRON if model == "nemotron" else PARAKEET_EOU, 1, "cuda", step_s=step_s,
                             cuda_graphs=graphs)
            r["step_s"], r["lookahead_steps"] = spec.step_s, spec.lookahead_steps
            r["streaming_cfg"] = {k: str(getattr(spec.model.encoder.streaming_cfg, k, None)) for k in
                                  ("chunk_size", "shift_size", "pre_encode_cache_size", "drop_extra_pre_encoded",
                                   "cache_drop_size", "valid_out_len")}
            t = time.perf_counter()
            if model == "nemotron":
                steps = record_steps(NemoStream(spec), audio, flush_steps=spec.lookahead_steps)
                text = pieces_text([p for ps, _ in steps for p in ps if p not in (EOU, EOB)])
                ms = [m for _, m in steps]
            else:
                res = quality.run_live(ParakeetEOU(NemoStream(spec)), audio, piece=int(spec.step_s * SR))
                text = " ".join(e["text"] for e in res["events"] if e["kind"] == "final")
                ms = [e["compute_ms"] for e in res["events"] if e["kind"] == "endpoint" and e["compute_ms"] > 0]
                r["endpoints"] = [round(e["audio_s"], 2) for e in res["events"] if e["kind"] == "endpoint"]
            r["wall_s"] = time.perf_counter() - t
            err, n = quality.word_errors(ref, text)
            r["wer"], r["text"] = 100 * err / max(n, 1), text
            r["step_ms_p50"] = float(np.percentile(ms[1:], 50)) if len(ms) > 1 else None
            if model == "nemotron" and step_s == 0.08 and graphs and not any(
                    v.get("model") == "nemotron" and "offline_wer" in v for v in out["variants"]):
                with tempfile.TemporaryDirectory() as d:
                    paths = []
                    for i, sg in enumerate(meta["segments"]):
                        paths.append(f"{d}/{i}.wav")
                        sf.write(paths[-1], audio[sg["start"]:sg["end"]], SR)
                    hyps = spec.model.transcribe(paths, batch_size=len(paths))
                    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
                    off = " ".join(getattr(h, "text", str(h)) for h in hyps)
                err, n = quality.word_errors(ref, off)
                r["offline_wer"], r["offline_text"] = 100 * err / max(n, 1), off
            del spec
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback

            r["error"] = f"{type(e).__name__}: {e}"
            r["traceback"] = traceback.format_exc()[-3000:]
        out["variants"].append(r)
        _log(json.dumps({k: v for k, v in r.items() if k not in ("text", "offline_text", "traceback")}))
    d = QUALITY / "diag"
    d.mkdir(parents=True, exist_ok=True)
    (d / "diag.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.function(image=image, cpu=0.25, memory=1024, timeout=5 * 3600, volumes={str(ROOT): vol})
def orchestrate(stage: str, systems: list[str], meetings: list[str], n_librispeech: int, gpu: str) -> str:
    from jevlite.voice.quality import system_dirs

    index = prepare.remote(meetings, n_librispeech)
    jobs = system_dirs(systems)
    dec = decode.with_options(gpu=gpu)
    if stage == "smoke":
        ids = [SMOKE_ID]
        full = sum(s["duration_s"] for s in index if s["id"] != SMOKE_ID) / 60
        dec.remote(jobs, ids, "smoke", smoke=True)
        return evaluate.remote("smoke", ids, full_minutes=full)
    if stage == "full":
        ids = [s["id"] for s in index if s["id"] != SMOKE_ID]
        dec.remote(jobs, ids, "full")
        return evaluate.remote("full", ids)
    if stage == "prepare":
        return f"{len(index)} streams prepared"
    if stage == "diag":
        res = diag.with_options(gpu=gpu).remote()
        return json.dumps([{k: v for k, v in r.items() if k not in ("text", "offline_text", "traceback")}
                           for r in res["variants"]], indent=1)
    raise ValueError(f"stage {stage!r}: smoke | full | prepare | diag")


@app.local_entrypoint()
def main(stage: str = "smoke", systems: str = ",".join(SYSTEMS), meetings: str = ",".join(MEETINGS),
         n_librispeech: int = 100, gpu: str = "L4"):
    from jevlite.voice.asr import parse_system

    names = [s for s in systems.split(",") if s.strip()]
    for s in names:
        parse_system(s)  # fail here, not after the data and the GPU
    report = orchestrate.remote(stage, names, [m for m in meetings.split(",") if m.strip()], n_librispeech, gpu)
    print(report)
    print("results: uv run --extra bench modal volume get jevlite-voice quality/ data/speed/voice/quality/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Start a stage on the deployed app; returns at once.")
    ap.add_argument("stage", choices=["smoke", "diag", "full", "prepare"])
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--meetings", default=",".join(MEETINGS))
    ap.add_argument("--n-librispeech", type=int, default=100)
    ap.add_argument("--gpu", default="L4")
    a = ap.parse_args()
    from jevlite.voice.asr import parse_system

    names = [x for x in a.systems.split(",") if x.strip()]
    for x in names:
        parse_system(x)
    fn = modal.Function.from_name(APP, "orchestrate")
    call = fn.spawn(a.stage, names, [x for x in a.meetings.split(",") if x.strip()], a.n_librispeech, a.gpu)
    print(f"started {a.stage}: call {call.object_id} on the deployed app {APP}")
    print(f"logs:    uv run --extra bench modal app logs {APP}")
    print(f"stop:    uv run --extra bench modal app stop {APP}")
    print("results: uv run --extra bench modal volume get jevlite-voice quality data/speed/voice")
