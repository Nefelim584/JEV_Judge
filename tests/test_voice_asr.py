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

    def ids_to_text(self, ids):
        return "".join(self.vocab[i] for i in ids).replace("▁", " ").strip()


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
    n = int(length[0]) // 160 + 1  # 10 ms hop, like NeMo's centred STFT
    x = input_signal[0]
    feats = torch.stack([torch.full((4,), float(x[min(i * 160, len(x) - 1)])) for i in range(n)], dim=1)
    return feats.unsqueeze(0), torch.tensor([n])


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
    assert s.step(np.zeros(STEP, np.float32)) == [0]
    assert s.step(np.zeros(STEP, np.float32)) == []
    assert s.step(np.zeros(STEP, np.float32)) == [1, 3]
    calls = s.model.encoder.calls
    assert [c["cache"] for c in calls] == [0, 1, 2]  # the encoder cache is carried
    assert calls[0]["shape"] == (1, 4, 20) and calls[0]["keep_all_outputs"] is False and calls[0]["drop"] == 2
    p = s.model.decoding.partials
    assert p[0] is None and p[1] is not None  # the hypothesis is carried
    s.reset()
    assert s.model.encoder.inits == 2 and s.hyps is None and s.seen == []


def test_nemo_stream_accepts_per_step_hypotheses():
    s = fake_stream([])
    assert s._new_tokens([0, 1]) == [0, 1]
    assert s._new_tokens([0, 1, 3]) == [3]  # cumulative
    assert s._new_tokens([5]) == [5]  # not an extension: this step's tokens


def test_split_markers_and_clean_text():
    assert A.split_markers(FakeTok(), [0, 1, 2]) == ([0, 1], A.EOU)
    assert A.split_markers(FakeTok(), [5, 4]) == ([5], A.EOB)
    assert A.split_markers(FakeTok(), []) == ([], "")
    assert A.clean_text(" hello  world<EOU>") == "hello world"


def test_parakeet_eou_endpoint_from_the_model():
    asr = A.ParakeetEOU(fake_stream([[0], [], [1], [2], [5]]))
    ev = asr.push(np.zeros(4 * STEP + 100, np.float32))
    assert [(e.kind, round(e.audio_s, 2), e.text, e.detail) for e in ev] == [
        ("partial", 0.08, "hello", ""), ("partial", 0.24, "hello world", ""),
        ("endpoint", 0.32, "", "eou"), ("final", 0.32, "hello world", "")]
    assert asr.stream.model.encoder.inits == 2  # state reset after the end of turn
    ev = asr.push(np.zeros(STEP - 100, np.float32))  # completes the 5th step
    assert [(e.kind, e.text) for e in ev] == [("partial", "yeah")]
    ev = asr.flush()
    assert [(e.kind, e.detail, e.text) for e in ev] == [("endpoint", "eos", ""), ("final", "", "yeah")]
    assert asr.flush() == []
    assert asr.compute_ms > 0


class FakeTranscriber:
    name = "fake"

    def __init__(self):
        self.accepted, self.turns, self.resets = 0, [], 0

    def accept(self, audio):
        self.accepted += len(audio)
        return "partial" if self.accepted == 10 * CHUNK else None

    def finalize(self, turn_audio):
        self.turns.append(turn_audio)
        return f"turn {len(self.turns)}"

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
    assert ev[2].text == "turn 1" and ev[2].audio_s == end.audio_s
    first = tr.turns[0]
    assert len(first) == int(end.audio_s * SR) - (s1 - int(0.3 * SR))  # preroll before the turn opened
    assert np.all(first[:int(0.3 * SR)] == 0) and np.all(first[int(0.3 * SR):int(0.3 * SR) + 100] == np.float32(0.1))
    # the second turn is still open at the end of the stream: flush ends it ("eos")
    assert ev[3].audio_s == pytest.approx(len(audio) // CHUNK * CHUNK / SR) and ev[4].text == "turn 2"
    assert tr.accepted == len(audio) // CHUNK * CHUNK


def test_separate_asr_keeps_only_max_turn_audio():
    tr = FakeTranscriber()
    asr = A.SeparateASR(tr, parse_setting("vad-300"), vad, max_turn_s=0.5, preroll_s=0.0)
    asr.push(np.full(2 * SR, 0.1, np.float32))
    kept = sum(len(c) for _, c in asr.history)
    assert kept <= int(0.5 * SR) + 2 * CHUNK
    (_, final) = asr.flush()
    assert len(tr.turns[0]) <= int(0.5 * SR) + 2 * CHUNK  # the turn is cut to the last max_turn_s


def test_nemotron_turns_flushes_the_lookahead():
    s = fake_stream([[0], [1], [3]], lookahead=2)
    tr = A.NemotronTurns(s)
    assert tr.accept(np.zeros(STEP + 10, np.float32)) == "hello"
    assert tr.accept(np.zeros(5, np.float32)) is None
    text = tr.finalize(np.zeros(0, np.float32))
    assert text == "hello world." and len(s.model.encoder.calls) == 1 + 1 + 2  # rest padded + 2 silent steps
    assert tr.ids == [] and s.model.encoder.inits == 2


def test_whisper_turns_decodes_the_turn():
    calls = {}

    class Model:
        def transcribe(self, audio, **kw):
            calls.update(kw, n=len(audio))
            return iter([SimpleNamespace(text=" Hello"), SimpleNamespace(text=" world. ")]), None

    tr = A.WhisperTurns(Model())
    assert tr.accept(np.zeros(10)) is None
    assert tr.finalize(np.zeros(SR, np.float32)) == "Hello world."
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
