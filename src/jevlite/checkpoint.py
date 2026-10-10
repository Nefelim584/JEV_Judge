"""Checkpoints of a training run: local files plus an optional mirror on the HF Hub (TODO Phase 6).

Layout, locally under ``runs_dir/<run>`` and on the Hub under ``<repo>/<run>``:

    <run>/config.yaml, metrics.jsonl, run.log   the run's files, refreshed at every checkpoint
    <run>/checkpoints/step-000123/state.pt      trainable weights, optimizer, scheduler, scaler, position
    <run>/final/                                the trained model (``save_final``)

Uploads run in one background thread, in order, so training does not wait for the network. After
each upload the checkpoints beyond ``keep`` are deleted and the repo history is squashed: files
deleted in a later commit still count against the Hub storage quota until the history is rewritten.
Squashing rewrites the whole branch, so run one training job per repo at a time.

The token comes from the environment (``HF_TOKEN``; on Kaggle from a notebook secret).
"""

from __future__ import annotations

import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
import yaml

from .log import logger

_STEP_RE = re.compile(r"/checkpoints/step-(\d+)/state\.pt$")


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Trainable parameters and all buffers, on the CPU. Frozen weights (the base encoder under LoRA)
    come back from the initial checkpoint, so they are not stored again."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def load_trainable_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Inverse of ``trainable_state_dict``: every key must exist; only frozen weights may be missing."""
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"checkpoint has unknown tensors: {result.unexpected_keys[:5]}")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = [k for k in result.missing_keys if k in trainable]
    if missing:
        raise ValueError(f"checkpoint lacks trainable tensors: {missing[:5]}")


class CheckpointStore:
    def __init__(self, runs_dir: str | Path, run: str, hub_repo: str | None = None, keep: int = 2):
        self.run = run
        self.dir = Path(runs_dir) / run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.repo = hub_repo
        self.keep = keep
        self._api = None
        self._pool: ThreadPoolExecutor | None = None
        self._pending: list[Future] = []
        if hub_repo:
            from huggingface_hub import HfApi

            self._api = HfApi()
            self._api.create_repo(hub_repo, private=True, exist_ok=True)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hub-upload")

    # ------------------------------------------------------------------ paths and listing

    def step_dir(self, step: int) -> Path:
        return self.dir / "checkpoints" / f"step-{step:06d}"

    def _repo_path(self, rel: str) -> str:
        return f"{self.run}/{rel}"

    def local_steps(self) -> list[int]:
        root = self.dir / "checkpoints"
        if not root.exists():
            return []
        return sorted(int(p.name[5:]) for p in root.glob("step-*") if (p / "state.pt").exists())

    def hub_steps(self) -> list[int]:
        if self._api is None:
            return []
        files = self._api.list_repo_files(self.repo)
        return sorted(int(m.group(1)) for f in files if f.startswith(f"{self.run}/") and (m := _STEP_RE.search(f)))

    # ------------------------------------------------------------------ save / resume

    def save(self, state: dict, step: int, run_files: Sequence[Path] = ()) -> Path:
        """Write ``state`` as ``step-N/state.pt``, keep the newest ``keep`` locally, and queue the upload
        of the checkpoint and of ``run_files`` (read now, so the upload sees a consistent snapshot)."""
        d = self.step_dir(step)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.pt.part"
        torch.save(state, tmp)
        tmp.replace(d / "state.pt")
        for old in self.local_steps()[: -self.keep]:
            shutil.rmtree(self.step_dir(old), ignore_errors=True)
        if self._pool is not None:
            from huggingface_hub import CommitOperationAdd

            ops = [CommitOperationAdd(self._repo_path(f"checkpoints/{d.name}/state.pt"), str(d / "state.pt"))]
            ops += [CommitOperationAdd(self._repo_path(Path(f).name), Path(f).read_bytes()) for f in run_files if Path(f).exists()]
            self._pending.append(self._pool.submit(self._upload, step, ops))
        logger.info("checkpoint step {} saved to {}", step, d)
        return d

    def _upload(self, step: int, ops: list) -> None:
        from huggingface_hub import CommitOperationDelete

        try:
            self._api.create_commit(self.repo, ops, commit_message=f"{self.run}: step {step}")
            old = self.hub_steps()[: -self.keep]
            if old:
                deletes = [CommitOperationDelete(self._repo_path(f"checkpoints/step-{s:06d}/")) for s in old]
                self._api.create_commit(self.repo, deletes, commit_message=f"{self.run}: drop steps {old}")
                self._api.super_squash_history(self.repo, commit_message="squash: keep only the current files")
            logger.info("checkpoint step {} uploaded to {}/{}", step, self.repo, self.run)
        except Exception as e:  # training goes on; the next checkpoint retries the upload
            logger.error("upload of step {} failed: {}: {}", step, type(e).__name__, e)

    def latest(self) -> Path | None:
        """The newest checkpoint, local or on the Hub (downloaded), or None for a fresh run."""
        local, hub = self.local_steps(), self.hub_steps()
        step = max(local + hub, default=None)
        if step is None:
            return None
        path = self.step_dir(step) / "state.pt"
        if not path.exists():
            from huggingface_hub import hf_hub_download

            logger.info("downloading checkpoint step {} from {}", step, self.repo)
            hf_hub_download(self.repo, self._repo_path(f"checkpoints/step-{step:06d}/state.pt"), local_dir=self.dir.parent)
        return path

    def restore_file(self, name: str) -> bool:
        """Fetch ``<run>/<name>`` from the Hub unless it exists locally (e.g. metrics.jsonl after a
        dead session), so logs continue where they stopped."""
        if (self.dir / name).exists() or self._api is None:
            return False
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            hf_hub_download(self.repo, self._repo_path(name), local_dir=self.dir.parent)
            return True
        except EntryNotFoundError:
            return False

    def upload_folder(self, folder: Path, rel: str) -> None:
        """Queue the upload of a whole folder (e.g. ``final/``) to ``<run>/<rel>``."""
        if self._pool is not None:
            self._pending.append(self._pool.submit(
                self._api.upload_folder, repo_id=self.repo, folder_path=str(folder),
                path_in_repo=self._repo_path(rel), commit_message=f"{self.run}: {rel}",
            ))

    def upload_files(self, files: Sequence[Path]) -> None:
        """Queue the upload of run files (read now) to ``<run>/<name>``."""
        if self._pool is not None:
            from huggingface_hub import CommitOperationAdd

            ops = [CommitOperationAdd(self._repo_path(Path(f).name), Path(f).read_bytes()) for f in files if Path(f).exists()]
            self._pending.append(self._pool.submit(self._api.create_commit, self.repo, ops, commit_message=f"{self.run}: run files"))

    def wait(self) -> None:
        """Block until every queued upload is done."""
        for f in self._pending:
            f.result()
        self._pending.clear()


# ---------------------------------------------------------------------- final model


def save_final(model, tokenizer, cfg: dict, out: str | Path) -> Path:
    """The trained model as a plain HF encoder (LoRA merged) plus ``heads.pt`` and ``jevlite.yaml``.

    Merging replaces ``model.encoder`` by the merged encoder, so call it after the last forward pass
    that needs the LoRA wrapper (predictions are the same either way).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
    model.encoder.save_pretrained(out / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(out / "encoder")
    heads = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(heads, out / "heads.pt")
    (out / "jevlite.yaml").write_text(yaml.safe_dump({"model": cfg["model"], "encoding": cfg["encoding"]}, sort_keys=False))
    return out


def load_final(path: str | Path, device: str | torch.device = "cpu", attn_implementation: str = "sdpa"):
    """``(model, tokenizer, cfg)`` from a ``save_final`` folder."""
    from transformers import AutoModel, AutoTokenizer

    from .model import JevLite

    path = Path(path)
    cfg = yaml.safe_load((path / "jevlite.yaml").read_text())
    encoder = AutoModel.from_pretrained(path / "encoder", attn_implementation=attn_implementation, dtype=torch.float32)
    m = cfg["model"]
    model = JevLite(encoder, hidden=encoder.config.hidden_size, trunk_dim=m["trunk_dim"], dropout=m["dropout"],
                    readout=m.get("readout", "marker"), decision_layers=m.get("decision_layers", 0))
    result = model.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"), strict=False)
    if result.unexpected_keys or any(not k.startswith("encoder.") for k in result.missing_keys):
        raise ValueError(f"heads.pt does not match the model: {result}")
    return model.to(device).eval(), AutoTokenizer.from_pretrained(path / "encoder"), cfg
