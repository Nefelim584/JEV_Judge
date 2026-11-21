from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jevlite.voice import asr as A
from jevlite.voice.endpoint import CHUNK, SR, Endpointer, parse_setting

STEP = int(A.STEP_S * SR)  # 1280 samples


class FakeTok:
    vocab = ["▁hello", "▁world", "<EOU>", ".", "<EOB>", "▁yeah"]

    def ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]


class FakeEncoder:
    def __init__(self):
        self.calls, self.inits = [], 0

    def get_initial_cache_state(self, batch_size):
        self.inits += 1
        return torch.zeros(1), torch.zeros(1), torch.zeros(1)

    def cache_aware_stream_step(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                                cache_last_channel_len, keep_all_outputs, drop_extra_pre_encoded):
        self.calls.append({"shape": tuple(processed_signal.shape), "cache": float(cache_last_channel[0]),
                           "keep_all_outputs": keep_all_outputs, "drop": drop_extra_pre_encoded})
        return processed_signal, processed_signal_length, cache_last_channel + 1, cache_last_time, cache_last_channel_len


class FakeDecoding:
    """Emits the scripted tokens per step, carrying the hypothesis (cumulative y_sequence, like NeMo)."""

    def __init__(self, script):
        self.script = list(script)
        self.partials = []

    def rnnt_decoder_predictions_tensor(self, enc, enc_len, return_hypotheses, partial_hypotheses):
        assert return_hypotheses
        self.partials.append(partial_hypotheses)
        prev = list(partial_hypotheses[0].y_sequence) if partial_hypotheses else []
        new = self.script.pop(0) if self.script else []
        return [SimpleNamespace(y_sequence=torch.tensor(prev + new, dtype=torch.long))]


def fake_preprocess(input_signal, length):
    """[B, L] audio → [B, 4, frames]: frame i = the sample at i × 10 ms (centred-STFT frame count)."""
    n = int(length[0]) // 160 + 1  # 10 ms hop, like NeMo's centred STFT
    idx = torch.tensor([min(i * 160, input_signal.shape[1] - 1) for i in range(n)])
    feats = input_signal[:, idx].unsqueeze(1).expand(-1, 4, -1)
    return feats, torch.full((input_signal.shape[0],), n)


def fake_stream(script, lookahead=1):
    model = SimpleNamespace(encoder=FakeEncoder(), decoding=FakeDecoding(script), tokenizer=FakeTok())
    spec = A.NemoSpec(name="fake", model=model, att_context_size=(70, lookahead), buffer_s=0.2, step_s=A.STEP_S,
                      window_stride_s=0.01, n_feat=4, drop_extra_pre_encoded=2, lookahead_steps=lookahead,
                      device="cpu")
    window = A.FeatureWindow(fake_preprocess, spec.buffer_s, spec.step_s, spec.window_stride_s, spec.n_feat)
    return A.NemoStream(spec, window)


def test_rechunker():
    r = A.Rechunker(4)
    assert r.push(np.arange(6)) and len(r.buf) == 2
    out = r.push(np.arange(6, 11))
    assert [c.tolist() for c in out] == [[4, 5, 6, 7]] and r.take_rest().tolist() == [8, 9, 10]
    assert len(r.buf) == 0


def test_feature_window_rolls_the_newest_frames_in():
    w = A.FeatureWindow(fake_preprocess, buffer_s=0.2, step_s=0.08, window_stride_s=0.01, n_feat=4)
    assert w.window.shape == (4, 20) and float(w.window[0, 0]) == pytest.approx(A.LOG_MEL_ZERO)
    out = w.update(np.full(STEP, 0.5, np.float32))
    assert out.shape == (4, 20)
    assert torch.all(out[:, :12] == A.LOG_MEL_ZERO) and torch.all(out[:, 12:] == 0.5)  # 8 new frames at the end
    out = w.update(np.full(STEP, 0.25, np.float32))
    assert torch.all(out[:, 4:12] == 0.5) and torch.all(out[:, 12:] == 0.25)
    with pytest.raises(ValueError):
        w.update(np.zeros(100, np.float32))
    w.reset()
    assert torch.all(w.window == A.LOG_MEL_ZERO)


def test_nemo_stream_carries_state_and_returns_new_tokens():
    s = fake_stream([[0], [], [1, 3]])
    assert s.step(np.zeros(STEP, np.float32)) == ["▁hello"]
    assert s.step(np.zeros(STEP, np.float32)) == []
    assert s.step(np.zeros(STEP, np.float32)) == ["▁world", "."] and s.last_ms > 0
    calls = s.model.encoder.calls
    assert [c["cache"] for c in calls] == [0, 1, 2]  # the encoder cache is carried
    assert calls[0]["shape"] == (1, 4, 20) and calls[0]["keep_all_outputs"] is False and calls[0]["drop"] == 2
    p = s.model.decoding.partials
    assert p[0] is None and p[1] is not None  # the hypothesis is carried
    s.reset()
    assert s.model.encoder.inits == 2 and s.batched.hyps is None and s.batched.seen == [[]]


def test_nemo_stream_accepts_per_step_hypotheses():
    b = fake_stream([]).batched
    assert b._new_tokens(0, [0, 1]) == [0, 1]
    assert b._new_tokens(0, [0, 1, 3]) == [3]  # cumulative
    assert b._new_tokens(0, [5]) == [5] and b.non_prefix_steps == 1  # not an extension: this step's tokens


def test_split_markers_and_pieces_text():
    assert A.split_markers(["▁hello", "▁world", "<EOU>"]) == (["▁hello", "▁world"], A.EOU)
    assert A.split_markers(["▁yeah", "<EOB>"]) == (["▁yeah"], A.EOB)
    assert A.split_markers([]) == ([], "")
    assert A.pieces_text(["▁hello", "▁wor", "ld", "."]) == "hello world."


def test_parakeet_eou_endpoint_from_the_model():
    asr = A.ParakeetEOU(fake_stream([[0], [], [1], [2], [5]]))
    inits = asr.stream.model.encoder.inits
    ev = asr.push(np.zeros(4 * STEP + 100, np.float32))
    assert [(e.kind, round(e.audio_s, 2), e.text, e.detail) for e in ev] == [
        ("partial", 0.08, "hello", ""), ("partial", 0.24, "hello world", ""),
        ("endpoint", 0.32, "", "eou"), ("final", 0.32, "hello world", "")]
    assert asr.stream.model.encoder.inits == inits + 1  # state reset after the end of turn
    ev = asr.push(np.zeros(STEP - 100, np.float32))  # completes the 5th step
    assert [(e.kind, e.text) for e in ev] == [("partial", "yeah")]
    ev = asr.flush()
    assert [(e.kind, e.detail, e.text) for e in ev] == [("endpoint", "eos", ""), ("final", "", "yeah")]
    assert asr.flush() == []
    assert asr.compute_ms > 0


class FakeTranscriber:
    name = "fake"
    lag_samples = 0

    def __init__(self):
        self.processed, self.turns, self.resets, self.compute_ms, self.finished = 0, [], 0, 0.0, 0

    def accept(self, audio):
        self.processed += len(audio)
        return "partial" if self.processed == 10 * CHUNK else None

    def finalize(self, turn_audio, until):
        self.turns.append(turn_audio)
        self.compute_ms += 5.0
        return f"turn {len(self.turns)}"

    def finish(self):
        self.finished += 1

    def reset(self):
        self.resets += 1


def speech_stream():
    """0.5 s silence, 1.0 s "speech", 1.0 s silence, 0.6 s speech, 0.2 s silence (chunk-aligned; < 300 ms: open at the end)."""
    def secs(s):
        return int(round(s * SR / CHUNK)) * CHUNK
    parts = [np.zeros(secs(0.5)), np.full(secs(1.0), 0.1), np.zeros(secs(1.0)), np.full(secs(0.6), 0.1),
             np.zeros(secs(0.2))]
    return np.concatenate(parts).astype(np.float32), [secs(0.5), secs(1.5), secs(2.5), secs(3.1)]


def vad(chunk):
    return 1.0 if float(np.abs(chunk).mean()) > 0.01 else 0.0


def test_separate_asr_endpoint_final_and_turn_audio():
    audio, (s1, e1, s2, e2) = speech_stream()
    tr = FakeTranscriber()
    asr = A.SeparateASR(tr, parse_setting("vad-300"), vad, preroll_s=0.3)
    assert asr.name == "fake+vad-300" and tr.resets == 1
    ev = []
    for i in range(0, len(audio), 1000):  # odd piece sizes, as from a socket
        ev += asr.push(audio[i:i + 1000])
    ev += asr.flush()
    kinds = [(e.kind, e.detail) for e in ev]
    assert kinds == [("partial", ""), ("endpoint", "silence"), ("final", ""), ("endpoint", "eos"), ("final", "")]
    end = ev[1]
    wait = end.audio_s - e1 / SR
    assert 0.3 <= wait < 0.3 + 2 * CHUNK / SR
    assert ev[2].text == "turn 1" and ev[2].audio_s == end.audio_s and ev[2].compute_ms == 5.0
    first = tr.turns[0]
    assert len(first) == int(end.audio_s * SR) - (s1 - int(0.3 * SR))  # preroll before the turn opened
    assert np.all(first[:int(0.3 * SR)] == 0) and np.all(first[int(0.3 * SR):int(0.3 * SR) + 100] == np.float32(0.1))
    # the second turn is still open at the end of the stream: flush ends it ("eos")
    assert ev[3].audio_s == pytest.approx(len(audio) // CHUNK * CHUNK / SR) and ev[4].text == "turn 2"
    assert tr.processed == len(audio) // CHUNK * CHUNK and tr.finished == 1


def test_separate_asr_keeps_only_max_turn_audio():
    tr = FakeTranscriber()
    asr = A.SeparateASR(tr, parse_setting("vad-300"), vad, max_turn_s=0.5, preroll_s=0.0)
    asr.push(np.full(2 * SR, 0.1, np.float32))
    kept = sum(len(c) for _, c in asr.history)
    assert kept <= int(0.5 * SR) + 2 * CHUNK
    (_, final) = asr.flush()
    assert len(tr.turns[0]) <= int(0.5 * SR) + 2 * CHUNK  # the turn is cut to the last max_turn_s


def test_nemotron_turns_final_waits_for_the_lookahead():
    # 80 ms steps: "hello" at step 1, "world" at 3, "." at 5; "yeah" at step 11 (0.88 s), after the cutoff
    # (endpoint ~0.54 s + 2 steps): it belongs to the next turn
    script = [[0], [], [1], [], [3], [], [], [], [], [], [5]]
    tr = A.NemotronTurns(fake_stream(script, lookahead=2), lag_steps=2)
    asr = A.SeparateASR(tr, parse_setting("vad-300"), lambda c: 1.0 if c.any() else 0.0, preroll_s=0.0)
    speech = np.full(int(0.2 * SR), 0.1, np.float32)
    ev = asr.push(np.concatenate([speech, np.zeros(int(0.9 * SR), np.float32)]))
    kinds = [(e.kind, e.text) for e in ev]
    assert ("partial", "hello") in kinds and ("partial", "hello world") in kinds
    end = next(e for e in ev if e.kind == "endpoint")
    final = next(e for e in ev if e.kind == "final")
    cutoff = end.audio_s + 2 * A.STEP_S
    assert final.audio_s == pytest.approx(cutoff, abs=A.STEP_S)  # waited 2 steps of real audio
    assert final.audio_s >= end.audio_s + 2 * A.STEP_S - 1e-9
    assert final.text == "hello world."
    assert [p for _, p in tr.tokens] == ["▁yeah"]  # waits for the next turn
    assert tr.lag_samples == 2 * STEP
    inits = tr.stepper.model.encoder.inits
    asr.push(np.zeros(int(0.5 * SR), np.float32))
    assert tr.stepper.model.encoder.inits == inits  # no reset at the turn: continuous decoding


def test_recorded_steps_replay_matches_live():
    script = [[0], [], [1], [], [3], [5], [], [0]]
    audio = np.concatenate([np.full(int(0.4 * SR), 0.1), np.zeros(int(0.4 * SR)), np.full(int(0.2 * SR), 0.1),
                            np.zeros(int(0.05 * SR))]).astype(np.float32)
    vad = lambda c: 1.0 if c.any() else 0.0  # noqa: E731

    def run(tr):
        asr = A.SeparateASR(tr, parse_setting("vad-300"), vad, preroll_s=0.0)
        ev = []
        for i in range(0, len(audio), 700):
            ev += asr.push(audio[i:i + 700])
        return [(e.kind, round(e.audio_s, 3), e.text, e.detail) for e in ev + asr.flush()]

    live = run(A.NemotronTurns(fake_stream(script, lookahead=1), lag_steps=1))
    rec = A.record_steps(fake_stream(script, lookahead=1), audio, flush_steps=1)
    assert len(rec) == int(np.ceil(len(audio) / STEP)) + 1
    replay = run(A.NemotronTurns(A.RecordedSteps(rec), lag_steps=1))
    assert replay == live and [k for k, *_ in live].count("final") == 2


def test_replay_vad():
    v = A.ReplayVAD(np.array([0.1, 0.9]))
    assert [v(None), v(None), v(None)] == [pytest.approx(0.1), pytest.approx(0.9), 0.0]
    v.reset()
    assert v(None) == pytest.approx(0.1)


def test_whisper_turns_decodes_the_turn():
    calls = {}

    class Model:
        def transcribe(self, audio, **kw):
            calls.update(kw, n=len(audio))
            return iter([SimpleNamespace(text=" Hello"), SimpleNamespace(text=" world. ")]), None

    tr = A.WhisperTurns(Model())
    assert tr.accept(np.zeros(10)) is None
    assert tr.finalize(np.zeros(SR, np.float32), until=0) == "Hello world." and tr.compute_ms > 0
    assert calls["n"] == SR and calls["beam_size"] == 1 and calls["language"] == "en"
    assert calls["condition_on_previous_text"] is False and calls["vad_filter"] is False


def test_parse_system():
    assert A.parse_system("parakeet-eou") == A.SystemSpec("parakeet-eou", "parakeet-eou", 1, None)
    assert A.parse_system("nemotron-r13+vad-500") == A.SystemSpec("nemotron-r13+vad-500", "nemotron", 13, "vad-500")
    assert A.parse_system("nemotron+turn-400-p0.5-m800").right_context == 1
    assert A.parse_system("whisper+vad-800") == A.SystemSpec("whisper+vad-800", "whisper", None, "vad-800")
    for bad in ("parakeet-eou+vad-500", "nemotron", "whisper-r1+vad-500", "nemotron-r2+vad-500", "whisper+vad",
                "kyutai"):
        with pytest.raises(ValueError):
            A.parse_system(bad)


def test_endpointer_close():
    ep = Endpointer(parse_setting("vad-500"))
    assert ep.close() is None
    for _ in range(10):
        ep.push(np.full(CHUNK, 0.1, np.float32), 1.0)
    e = ep.close()
    assert e.reason == "eos" and e.sample == 10 * CHUNK and not ep.in_turn


def test_model_pool_builds_streams(monkeypatch):
    loads = []

    def fake_load(name, right, device):
        loads.append((name, right))
        return fake_stream([[0]], lookahead=right).spec

    monkeypatch.setattr(A, "load_nemo", fake_load)
    monkeypatch.setattr(A, "nemo_preprocessor", lambda spec: fake_preprocess)
    pool = A.ModelPool(device="cpu")
    assert pool.nemo(A.PARAKEET_EOU, 1) is pool.nemo(A.PARAKEET_EOU, 1)  # loaded once
    a = pool.stream("parakeet-eou")
    b = pool.stream("parakeet-eou")
    assert isinstance(a, A.ParakeetEOU) and a.stream is not b.stream  # one state per stream
    assert loads == [(A.PARAKEET_EOU, 1)]
    ev = a.push(np.zeros(STEP, np.float32))
    assert [e.kind for e in ev] == ["partial"]


# ------------------------------------------------------------------------------------------- batching


class StateEncoder:
    """Real cache shapes ([layers, B, …], lengths [B]); the cache counts each slot's steps since its reset and
    the output carries it, so the decoder below depends on the slot's state, not only on its audio."""

    def get_initial_cache_state(self, batch_size):
        return torch.zeros(1, batch_size, 1), torch.zeros(1, batch_size, 1), torch.zeros(batch_size, dtype=torch.long)

    def cache_aware_stream_step(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                                cache_last_channel_len, keep_all_outputs, drop_extra_pre_encoded):
        c1 = cache_last_channel + 1
        enc = torch.stack([processed_signal[:, 0, -1], c1[0, :, 0]], dim=1).unsqueeze(-1)  # [B, 2, 1]
        return enc, processed_signal_length, c1, cache_last_time, cache_last_channel_len + 1


class StateDecoding:
    """Per slot: value v ≥ 1 in the newest feature frame → token (v - 1 + steps since reset) mod 6, appended
    to the slot's hypothesis (a ``None`` hypothesis starts empty, as NeMo's batched greedy decoder)."""

    def rnnt_decoder_predictions_tensor(self, enc, enc_len, return_hypotheses, partial_hypotheses):
        out = []
        for b in range(enc.shape[0]):
            h = partial_hypotheses[b] if partial_hypotheses is not None else None
            prev = list(h.y_sequence) if h is not None else []
            v, c = int(round(float(enc[b, 0, 0]))), int(enc[b, 1, 0])
            out.append(SimpleNamespace(y_sequence=torch.tensor(prev + ([(v - 1 + c) % 6] if v >= 1 else []),
                                                                 dtype=torch.long)))
        return out


def state_spec(lookahead=1):
    model = SimpleNamespace(encoder=StateEncoder(), decoding=StateDecoding(), tokenizer=FakeTok())
    return A.NemoSpec(name="fake", model=model, att_context_size=(70, lookahead), buffer_s=0.2, step_s=A.STEP_S,
                      window_stride_s=0.01, n_feat=4, drop_extra_pre_encoded=2, lookahead_steps=lookahead,
                      device="cpu")


def random_streams(seed=0, n=5):
    """Streams of steps with a constant value 0–6 each (0 = silence), lengths not a multiple of a step."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 25))
        a = np.repeat(rng.integers(0, 7, size=k).astype(np.float32), STEP)
        out.append(a[:len(a) - int(rng.integers(0, STEP))])
    return out


def test_batched_feature_window_matches_single_windows():
    streams = random_streams(1, 3)
    bw = A.BatchedFeatureWindow(fake_preprocess, 0.2, 0.08, 0.01, 4, batch=3)
    singles = [A.FeatureWindow(fake_preprocess, 0.2, 0.08, 0.01, 4) for _ in streams]
    for k in range(3):
        chunks = np.stack([s[k * STEP:(k + 1) * STEP] for s in streams])
        out = bw.update(chunks)
        for b, w in enumerate(singles):
            assert torch.equal(out[b], w.update(chunks[b]))
    bw.reset([1])
    assert torch.all(bw.windows[1] == A.LOG_MEL_ZERO) and not torch.all(bw.windows[0] == A.LOG_MEL_ZERO)


def test_stream_steps():
    a = np.arange(5, dtype=np.float32)
    assert A.stream_steps(a, 2).tolist() == [[0, 1], [2, 3], [4, 0]]
    assert A.stream_steps(a, 2, flush_steps=1).shape == (4, 2)
    assert A.stream_steps(a, 2, pad_last=False, flush_steps=1).tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_record_batched_equals_single_streams(batch):
    """B1 timelines: any batch size (2 = slots refilled as streams end) gives each stream's single-stream
    timeline; the compute is shared among the busy slots."""
    streams = random_streams(2, 5)
    spec = state_spec()
    singles = [[p for p, _ in A.record_steps(A.NemoStream(spec, A.FeatureWindow(fake_preprocess, 0.2, 0.08, 0.01, 4)),
                                             s, flush_steps=1)] for s in streams]
    timelines, info = A.record_batched(A.BatchedNemo(spec, batch, preprocess=fake_preprocess), streams, flush_steps=1)
    assert [[p for p, _ in t] for t in timelines] == singles
    assert any(p for t in singles for p in t)  # the fake emits something
    assert info["batch"] == batch and info["steps"] >= max(len(t) for t in singles)
    assert all(ms > 0 for t in timelines for _, ms in t)


def test_record_batched_parakeet_resets_like_live():
    """A1: the slot is reset after a step with <EOU> / <EOB>; replayed through ParakeetEOU the events equal the
    live ParakeetEOU's on each stream."""
    from jevlite.voice import quality as Q

    streams = random_streams(3, 4)
    spec = state_spec()
    live = []
    for s in streams:
        asr = A.ParakeetEOU(A.NemoStream(spec, A.FeatureWindow(fake_preprocess, 0.2, 0.08, 0.01, 4)))
        live.append([(e["kind"], round(e["audio_s"], 3), e["text"], e["detail"])
                     for e in Q.run_live(asr, s)["events"]])
    assert any(k == "endpoint" for ev in live for k, *_ in ev)  # resets happen
    res, info = Q.decode_batched(spec, streams, "parakeet", batch=2, preprocess=fake_preprocess)
    got = [[(e["kind"], round(e["audio_s"], 3), e["text"], e["detail"]) for e in r["events"]] for r in res]
    assert got == live and all(r["batch"] == 2 for r in res)


def test_decode_batched_nemotron_result_format():
    from jevlite.voice import quality as Q

    streams = random_streams(4, 3)
    res, _ = Q.decode_batched(state_spec(), streams, "nemotron", preprocess=fake_preprocess)
    r = res[0]
    assert set(r) >= {"steps", "compute_ms", "audio_s", "step_s", "lag_steps", "batch"} and r["batch"] == 3
    assert len(r["steps"]) == -(-len(streams[0]) // STEP) + 1 and r["audio_s"] == len(streams[0]) / SR
