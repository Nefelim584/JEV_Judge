"""The training loop on a tiny random ModernBERT (CPU, no downloads besides the cached tokenizer)."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from jevlite.checkpoint import CheckpointStore, load_final, load_trainable_state_dict, trainable_state_dict
from jevlite.config import load_config
from jevlite.data.unified import Record, load_predictions, write_jsonl
from jevlite.model import JevLite
from jevlite.data.public.base import stable_hash
from jevlite.train import eval_subset, length_grouped_batches, rank_batches, train

HIDDEN = 32


def test_length_grouped_batches_are_deterministic_full_and_grouped():
    lengths = [(i * 37) % 101 for i in range(1000)]
    a = length_grouped_batches(lengths, 8, seed=1, epoch=0, window=4)
    assert a == length_grouped_batches(lengths, 8, seed=1, epoch=0, window=4)
    assert a != length_grouped_batches(lengths, 8, seed=1, epoch=1, window=4)
    assert all(len(b) == 8 for b in a) and len(a) == 125
    flat = [i for b in a for i in b]
    assert len(set(flat)) == len(flat) == 1000
    # within a batch the lengths are close: much less spread than random batches
    spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in a) / len(a)
    assert spread < 30


def test_rank_batches_split_each_step_between_ranks():
    batches = [[i] for i in range(40)]
    world, accum, steps = 2, 2, 10
    ranks = [rank_batches(batches, 0, steps, world, accum, r) for r in range(world)]
    assert sorted(b[0] for r in ranks for b in r) == list(range(40))
    # step 3, micro-step 1, rank 1 → global batch (3·2 + 1)·2 + 1
    assert ranks[1][3 * accum + 1] == [15]
    # resuming at step 4 gives the tail of the same sequence
    assert rank_batches(batches, 4, steps, world, accum, 0) == ranks[0][4 * accum:]


@pytest.fixture
def tiny(tokenizer):
    from transformers import ModernBertConfig, ModernBertModel

    def make():
        torch.manual_seed(0)
        enc = ModernBertModel(ModernBertConfig(
            vocab_size=len(tokenizer), hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, pad_token_id=tokenizer.pad_token_id, reference_compile=False,
        ))
        return JevLite(enc, hidden=HIDDEN, trunk_dim=8, dropout=0.0)

    return make


def _records(split: str, n: int) -> list[Record]:
    out = []
    for i in range(n):
        state = f"Ticket {i}: the order {i % 7} arrived {'late' if i % 2 else 'on time'}."
        if i % 3 == 0:
            out.append(Record(id=f"t:{split}:{i}", source="t", split=split, state=state, type="bool", prompt="The order was late.", target=float(i % 2)))
        elif i % 3 == 1:
            out.append(Record(id=f"t:{split}:{i}", source="t", split=split, state=state, type="choice", prompt="When did it arrive?", candidates=["late", "on time", "never"], target=0 if i % 2 else 1))
        else:
            out.append(Record(id=f"t:{split}:{i}", source="t", split=split, state=state, type="score", prompt="How late?", candidates=["not", "a bit", "very"], target=2 if i % 2 else 0))
    return out


@pytest.fixture
def setup(tmp_path):
    mix = tmp_path / "mix" / "stage_a"
    write_jsonl(mix / "train.e0.jsonl", _records("train", 48))
    write_jsonl(mix / "test_in.jsonl", _records("test_in", 9))
    write_jsonl(mix / "test_ood.jsonl", _records("test_ood", 6))
    write_jsonl(mix / "dev.jsonl", _records("dev", 12))
    cfg = load_config("apple_silicon")
    cfg["runtime"] |= {"device": "cpu", "precision": "fp32", "attn_implementation": "sdpa", "optimizer": "adamw"}
    cfg["tuning"]["mode"] = "full"
    cfg["model"] |= {"trunk_dim": 8, "dropout": 0.0}  # as the tiny model, so load_final rebuilds it
    cfg["encoding"]["max_len"] = 64
    t = cfg["train"]
    t |= {"per_device_batch_size": 4, "effective_batch_size": 8, "num_workers": 0, "log_every_steps": 1,
          "eval_every_steps": 2, "eval_max_records": 5, "eval_batch_size": 4, "gradient_checkpointing": False}
    t["stage_a"] |= {"epochs": 1, "max_len": 64}
    t["checkpoint"] = {"every_minutes": 10_000, "keep": 2, "hub_repo": None}
    return cfg, tmp_path


def _run(cfg, root, tokenizer, model, run, **kw):
    return train(cfg, data_root=root / "mix", runs_dir=root / "runs", run_name=run, model=model, tokenizer=tokenizer, **kw)


def test_train_resume_matches_uninterrupted_run(setup, tokenizer, tiny):
    cfg, root = setup
    full = tiny()
    r = _run(cfg, root, tokenizer, full, "full")
    assert r == {"status": "done", "step": 6, "total_steps": 6, "run_dir": str(root / "runs" / "full")}

    first = _run(cfg, root, tokenizer, tiny(), "resumed", max_steps=2)
    assert first["status"] == "max_steps" and first["step"] == 2
    resumed = tiny()
    second = _run(cfg, root, tokenizer, resumed, "resumed")
    assert second["status"] == "done" and second["step"] == 6
    for (name, a), b in zip(full.named_parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6), name


def test_train_writes_logs_predictions_and_final_model(setup, tokenizer, tiny):
    cfg, root = setup
    model = tiny()
    _run(cfg, root, tokenizer, model, "run")
    run = root / "runs" / "run"
    metrics = (run / "metrics.jsonl").read_text().splitlines()
    evals = [json.loads(line) for line in metrics if '"phase": "eval"' in line]
    assert evals and {e["split"] for e in evals} == {"dev"}  # the tests never take part in the periodic eval
    assert any('"loss/bool"' in line for line in metrics)
    preds = load_predictions(run / "preds.jsonl")
    assert len(preds) == 15  # all of test_in + test_ood, not the eval subset
    assert [p.name for p in sorted((run / "checkpoints").iterdir())] == ["step-000006"]

    loaded, _, _ = load_final(run / "final")
    batch_model = copy.deepcopy(model).eval()
    for a, b in zip(batch_model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_store_keeps_newest_and_trainable_state_skips_frozen(tmp_path, tiny):
    store = CheckpointStore(tmp_path, "r", keep=2)
    for step in (1, 2, 3):
        store.save({"step": step}, step)
    assert store.local_steps() == [2, 3]
    assert store.latest() == store.step_dir(3) / "state.pt"

    model = tiny()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    state = trainable_state_dict(model)
    assert not any(k.startswith("encoder.") for k in state) and "log_temperature" in state
    target = tiny()
    for p in target.encoder.parameters():
        p.requires_grad_(False)
    load_trainable_state_dict(target, state)
    with pytest.raises(ValueError):  # a trainable encoder needs its weights in the checkpoint
        load_trainable_state_dict(tiny(), state)
    with pytest.raises(ValueError):
        load_trainable_state_dict(model, {"nope": torch.zeros(1)})


class FakeHub:
    """The part of ``HfApi`` that ``CheckpointStore`` uses, backed by a dict."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.squashes = 0

    def create_repo(self, *a, **kw):
        pass

    def list_repo_files(self, repo):
        return sorted(self.files)

    def create_commit(self, repo, ops, commit_message=""):
        from huggingface_hub import CommitOperationAdd

        for op in ops:
            if isinstance(op, CommitOperationAdd):
                src = op.path_or_fileobj
                self.files[op.path_in_repo] = src if isinstance(src, bytes) else open(src, "rb").read()
            else:
                prefix = op.path_in_repo
                self.files = {k: v for k, v in self.files.items() if not k.startswith(prefix)}

    def super_squash_history(self, repo, commit_message=""):
        self.squashes += 1


def test_store_mirrors_to_hub_prunes_and_resumes_from_it(tmp_path, monkeypatch):
    import huggingface_hub

    hub = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: hub)
    store = CheckpointStore(tmp_path / "a", "run", hub_repo="me/runs", keep=2)
    (store.dir / "metrics.jsonl").write_text('{"phase": "train"}\n')
    for step in (10, 20, 30):
        store.save({"step": step}, step, [store.dir / "metrics.jsonl"])
    store.wait()
    assert store.hub_steps() == [20, 30] and hub.squashes == 1
    assert hub.files["run/metrics.jsonl"] == b'{"phase": "train"}\n'

    def download(repo, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(hub.files[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    fresh = CheckpointStore(tmp_path / "b", "run", hub_repo="me/runs", keep=2)  # a new session: empty disk
    path = fresh.latest()
    assert path == fresh.step_dir(30) / "state.pt" and torch.load(path)["step"] == 30
    assert fresh.restore_file("metrics.jsonl") and (fresh.dir / "metrics.jsonl").exists()


def test_eval_subset_per_primitive():
    recs = _records("dev", 30)  # 10 bool, 10 choice, 10 score
    sub = eval_subset(recs, {"bool": 3, "score": 20}, 42)
    assert Counter(r.type for r in sub) == {"bool": 3, "choice": 10, "score": 10}
    assert eval_subset(recs, {"bool": 3}, 42) == eval_subset(recs, {"bool": 3}, 42)
    assert [r.id for r in eval_subset(recs, 5, 42)] == [r.id for r in sorted(recs, key=lambda r: stable_hash(42, "eval", r.id))[:5]]


def test_build_model_from_a_stage_checkpoint(setup, tokenizer, tiny, tmp_path, monkeypatch):
    """``model.init: checkpoint`` (Stage B from Stage A): encoder and heads come from a save_final folder."""
    from jevlite.checkpoint import save_final
    from jevlite.device import resolve_runtime
    from jevlite.model import build_model, resolve_checkpoint

    cfg, _ = setup
    trained = tiny()
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(torch.randn_like(p))          # not the fresh init: equality below means "loaded"
    path = save_final(trained, tokenizer, cfg, tmp_path / "stage_a" / "final")
    cfg = {**cfg, "model": {**cfg["model"], "init": "checkpoint", "init_checkpoint": str(path)},
           "train": {**cfg["train"], "gradient_checkpointing": False}}
    model = build_model(cfg, resolve_runtime(cfg), mode="full")
    for (name, a), b in zip(trained.state_dict().items(), model.state_dict().values()):
        assert torch.equal(a, b), name

    bad = {**cfg, "model": {**cfg["model"], "trunk_dim": 16}}
    with pytest.raises(ValueError, match="trunk_dim"):
        build_model(bad, resolve_runtime(bad), mode="full")

    import huggingface_hub
    seen = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, allow_patterns, token=None: seen.update(repo=repo, patterns=allow_patterns) or "/cache/snap")
    assert resolve_checkpoint("hub:Nefelim584/jevlite-runs/stage_a-full-s42/final") == Path("/cache/snap/stage_a-full-s42/final")
    assert seen == {"repo": "Nefelim584/jevlite-runs", "patterns": ["stage_a-full-s42/final/*"]}


def test_heads_get_their_own_learning_rate(setup, tokenizer, tiny):
    """``train.lr.heads`` (todo, "Stage A — head learning rate"): trunk + heads in a second AdamW group, logged
    as ``lr_heads``; ``head_drift`` shows they move. Unset, one group as in Stage A."""
    cfg, root = setup
    one = _run(copy.deepcopy(cfg), root, tokenizer, tiny(), "one_group", max_steps=4)
    two_cfg = copy.deepcopy(cfg)
    two_cfg["train"]["lr"]["heads"] = 100 * two_cfg["train"]["lr"]["full"]
    two = _run(two_cfg, root, tokenizer, tiny(), "two_groups", max_steps=4)

    def last_train(run):
        rows = [json.loads(line) for line in (Path(run["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
        return [r for r in rows if r.get("kind", r.get("phase")) == "train" or "head_drift" in r][-1]

    a, b = last_train(one), last_train(two)
    assert "lr_heads" not in a and b["lr_heads"] == pytest.approx(100 * b["lr"])
    assert b["head_drift"] > 5 * a["head_drift"] > 0


def test_resume_with_a_separate_head_group_matches_uninterrupted_run(setup, tokenizer, tiny):
    """Stage B spans several Kaggle sessions: the two-group optimizer state must resume exactly."""
    cfg, root = setup
    cfg = copy.deepcopy(cfg)
    cfg["train"]["lr"]["heads"] = 10 * cfg["train"]["lr"]["full"]
    full = tiny()
    assert _run(cfg, root, tokenizer, full, "g_full")["status"] == "done"
    assert _run(cfg, root, tokenizer, tiny(), "g_resumed", max_steps=2)["step"] == 2
    resumed = tiny()
    assert _run(cfg, root, tokenizer, resumed, "g_resumed")["step"] == 6
    for (name, a), b in zip(full.named_parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6), name
