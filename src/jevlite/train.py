"""The training loop of every stage (TODO Phase 6), shared by both stacks, the notebooks and
``scripts/train.py``.

- **Data:** the mix files ``<data_root>/<mix>/train.e{epoch}.jsonl``; micro-batches of records of
  similar length (``length_grouped_batches``), because random batches pad 188 real tokens per record
  to ~480. The batch order depends only on (seed, epoch), so a resumed run sees the same batches.
- **Steps:** one optimizer step = ``world_size × grad_accum`` micro-batches; ``grad_accum`` keeps
  ``train.effective_batch_size`` fixed across stacks. AMP and GradScaler come from ``device.py``.
- **DDP:** under ``torchrun`` every process trains on its share of each step; only rank 0 logs,
  evaluates and writes checkpoints. Rank 0 decides when to evaluate, checkpoint or stop, and
  broadcasts the decision, so all ranks leave the loop together.
- **Checkpoints:** every ``train.checkpoint.every_minutes`` and at the end (``checkpoint.py``,
  optionally mirrored to the HF Hub). ``max_minutes`` stops early with a checkpoint (Kaggle sessions
  end after 12 h); the next call with the same run name resumes from it.
- **Eval:** every ``train.eval_every_steps`` on a fixed subset of ``test_in`` / ``test_ood``; after
  the last step, predictions on all of both (``preds.jsonl``, the ``scripts/eval.py`` format) and the
  final model (``final/``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import random
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import yaml

from .checkpoint import CheckpointStore, load_trainable_state_dict, save_final, trainable_state_dict
from .collate import TYPE_IDS, group_logits
from .config import grad_accum_steps, seed_everything
from .data.loader import JsonlRecords, TrainCollator
from .data.public.base import stable_hash
from .data.unified import Record, load_records, write_jsonl
from .device import Runtime, autocast, make_grad_scaler, make_optimizer, resolve_runtime
from .encoding import text_encoder_from_config
from .evaluation import TYPES, primitive_metrics
from .inference import Predictor
from .log import logger
from .losses import LossConfig, jev_loss
from .model import JevLite, build_model, question_probs
from .reports.training import TrainingLog

EVAL_SPLITS = ("test_in", "test_ood")


# ---------------------------------------------------------------------------- data order


def length_grouped_batches(lengths: Sequence[int], batch_size: int, seed: int, epoch: int, window: int = 64) -> list[list[int]]:
    """Micro-batches of records of similar length, in an order fixed by (seed, epoch).

    Shuffle all indices, sort each run of ``window × batch_size`` by length, cut it into batches and
    shuffle the batches. Lengths only need to rank records, so the JSON line length is enough. An
    incomplete last batch is dropped.
    """
    rng = random.Random(stable_hash(seed, "batches", epoch))
    order = list(range(len(lengths)))
    rng.shuffle(order)
    span = window * batch_size
    batches: list[list[int]] = []
    for start in range(0, len(order), span):
        chunk = sorted(order[start:start + span], key=lambda i: lengths[i])
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    batches = [b for b in batches if len(b) == batch_size]
    rng.shuffle(batches)
    return batches


def rank_batches(batches: list[list[int]], step_from: int, steps: int, world: int, accum: int, rank: int) -> list[list[int]]:
    """This rank's micro-batches for optimizer steps ``step_from … steps − 1``: step s, micro-step a uses
    global batch ``(s × accum + a) × world + rank``."""
    return [batches[(s * accum + a) * world + rank] for s in range(step_from, steps) for a in range(accum)]


def eval_subset(records: list[Record], n: int | None, seed: int) -> list[Record]:
    """A fixed subset (stable hash of the id), the same in every run and session."""
    if n is None or len(records) <= n:
        return records
    return sorted(records, key=lambda r: stable_hash(seed, "eval", r.id))[:n]


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------- metrics


def batch_correct(grouped: torch.Tensor, target: torch.Tensor, question_type: torch.Tensor) -> torch.Tensor:
    """Per question: is the top answer the target's top answer (Bool: the 0.5 side)?"""
    probs = question_probs(grouped.detach(), question_type)
    is_bool = question_type == TYPE_IDS["bool"]
    multi = probs.argmax(-1) == target.argmax(-1)
    return torch.where(is_bool, (probs[:, 0] > 0.5) == (target[:, 0] > 0.5), multi)


def evaluate_records(model: JevLite, text_encoder, rt: Runtime, splits: dict[str, list[Record]], batch_size: int) -> dict[str, dict[str, float]]:
    """``{split: {"accuracy/bool": …, "nll/bool": …, "ece/bool": …, …}}``. Leaves the model in eval mode."""
    predictor = Predictor(model, text_encoder, amp_dtype=rt.amp_dtype)
    out = {}
    for split, records in splits.items():
        preds = predictor.predict_records(records, batch_size)
        row = {}
        for prim in TYPES:
            pairs = [(r, p) for r, p in zip(records, preds) if r.type == prim]
            if pairs:
                m = primitive_metrics(prim, [r for r, _ in pairs], [p for _, p in pairs])
                row |= {f"{k}/{prim}": m[k] for k in ("accuracy", "nll", "ece") if k in m}
        out[split] = row
    return out


# ---------------------------------------------------------------------------- the loop


@dataclasses.dataclass
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def broadcast_flags(self, *flags: bool, device) -> list[bool]:
        if self.world == 1:
            return list(flags)
        t = torch.tensor([float(f) for f in flags], device=device)
        dist.broadcast(t, 0)
        return [bool(v) for v in t.tolist()]

    def broadcast_object(self, obj):
        if self.world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, 0)
        return box[0]


def train(
    cfg: dict,
    *,
    stage: str = "stage_a",
    data_root: str | Path = "data/mix",
    runs_dir: str | Path = "runs",
    run_name: str,
    hub_repo: str | None = None,
    max_steps: int | None = None,
    max_minutes: float | None = None,
    meta: dict | None = None,
    dist_info: Dist | None = None,
    model: JevLite | None = None,
    tokenizer=None,
) -> dict:
    """Train (or resume) one run. Returns ``{"status": "done" | "paused" | "max_steps", "step": …}``.

    ``model`` / ``tokenizer`` default to ``build_model`` / the base tokenizer (tests pass tiny ones).
    """
    d = dist_info or Dist()
    t = cfg["train"]
    stage_cfg = t[stage]
    rt = resolve_runtime(cfg)
    if rt.device.type == "cuda" and d.world > 1:
        rt = dataclasses.replace(rt, device=torch.device("cuda", d.local_rank))
    seed_everything(cfg["seed"] + d.rank)

    mix_name = Path(stage_cfg["mix"]).stem
    mix_dir = Path(data_root) / mix_name
    n_epochs = int(stage_cfg.get("epochs", 1))
    files = [mix_dir / f"train.e{e}.jsonl" for e in range(n_epochs)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"mix files missing: {missing}")

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    if model is None:
        model = build_model(cfg, rt)
    model = model.to(rt.device)
    text_encoder = text_encoder_from_config(cfg, tokenizer, max_len=stage_cfg["max_len"])
    collator = TrainCollator(text_encoder, seed=cfg["seed"])
    loss_cfg = LossConfig.from_config(cfg)

    accum = grad_accum_steps(cfg, d.world)
    micro = t["per_device_batch_size"]
    steps_per_epoch = [_count_lines(f) // (micro * d.world * accum) for f in files]
    total_steps = sum(steps_per_epoch)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = t["lr"][cfg["tuning"]["mode"]]
    optimizer = make_optimizer(params, rt, lr=lr, weight_decay=t["weight_decay"])
    from transformers import get_scheduler

    scheduler = get_scheduler(t.get("schedule", "linear"), optimizer, num_warmup_steps=int(t["warmup_ratio"] * total_steps), num_training_steps=total_steps)
    scaler = make_grad_scaler(rt)

    ck = t.get("checkpoint", {})
    store = CheckpointStore(runs_dir, run_name, hub_repo if d.main else None, keep=int(ck.get("keep", 2)))
    run_dir = store.dir

    # ---- resume
    start_epoch, start_step, gstep = 0, 0, 0
    path = store.latest() if d.main else None
    path = d.broadcast_object(str(path) if path else None)
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_step, gstep = state["epoch"], state["step_in_epoch"], state["step"]
        logger.info("resumed {} from step {} (epoch {}, step {} of {})", run_name, gstep, start_epoch, start_step, steps_per_epoch[start_epoch] if start_epoch < n_epochs else 0)
        del state
    if d.main:
        for name in ("metrics.jsonl", "run.log"):
            store.restore_file(name)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"stage": stage, "meta": meta or {}, "cfg": cfg}, sort_keys=False))
    log = TrainingLog(run_dir, run=run_name, meta={"stage": stage, "mix": mix_name, "total_steps": total_steps, **(meta or {})}) if d.main else None

    eval_sets: dict[str, list[Record]] = {}
    if d.main:
        for split in EVAL_SPLITS:
            p = mix_dir / f"{split}.jsonl"
            if p.exists():
                eval_sets[split] = load_records(p)
    eval_small = {s: eval_subset(r, t.get("eval_max_records"), cfg["seed"]) for s, r in eval_sets.items()}
    eval_bs = int(t.get("eval_batch_size", micro * 4))

    net = model
    if d.world > 1:
        from torch.nn.parallel import DistributedDataParallel

        net = DistributedDataParallel(model, device_ids=[d.local_rank] if rt.device.type == "cuda" else None)

    def run_files() -> list[Path]:
        return [run_dir / n for n in ("config.yaml", "metrics.jsonl", "run.log")]

    def checkpoint(epoch: int, step_in_epoch: int) -> None:
        if step_in_epoch >= steps_per_epoch[epoch]:
            epoch, step_in_epoch = epoch + 1, 0
        store.save({
            "model": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "step_in_epoch": step_in_epoch, "step": gstep,
        }, gstep, run_files())

    def run_eval() -> None:
        for split, row in evaluate_records(model, text_encoder, rt, eval_small, eval_bs).items():
            log.log("eval", gstep, split=split, **row)
            logger.info("eval step {} {}: {}", gstep, split, " ".join(f"{k} {v:.3f}" for k, v in sorted(row.items())))
        model.train()

    log_every, eval_every = int(t.get("log_every_steps", 50)), int(t.get("eval_every_steps", 1000))
    every_s = 60 * float(ck.get("every_minutes", 40))
    t0 = last_ckpt = time.monotonic()
    window: dict[str, list[float]] = {}
    status = "done"
    model.train()
    for epoch in range(start_epoch, n_epochs):
        collator.set_epoch(epoch)
        data = JsonlRecords(files[epoch])
        batches = length_grouped_batches([len(line) for line in data.lines], micro, cfg["seed"], epoch)
        s_from = start_step if epoch == start_epoch else 0
        mine = rank_batches(batches, s_from, steps_per_epoch[epoch], d.world, accum, d.rank)
        loader = torch.utils.data.DataLoader(
            data, batch_sampler=mine, collate_fn=collator, num_workers=int(t.get("num_workers", 2)),
            persistent_workers=False, prefetch_factor=4 if t.get("num_workers", 2) else None,
        )
        it = iter(loader)
        for s in range(s_from, steps_per_epoch[epoch]):
            for a in range(accum):
                batch, target = next(it)
                batch, target = batch.to(rt.device), target.to(rt.device)
                sync = net.no_sync() if d.world > 1 and a < accum - 1 else contextlib.nullcontext()
                with sync:
                    with autocast(rt):
                        logits = net(batch, apply_temperature=False)
                    grouped = group_logits(logits.float(), batch)
                    loss, parts = jev_loss(grouped, target, batch, loss_cfg)
                    scaler.scale(loss / accum).backward()
                if d.main:
                    window.setdefault("loss", []).append(loss.item())
                    correct = batch_correct(grouped, target, batch.question_type)
                    for prim, v in parts.items():
                        rows = batch.question_type == TYPE_IDS[prim]
                        window.setdefault(f"loss/{prim}", []).append(v.item())
                        window.setdefault(f"accuracy/{prim}", []).append(correct[rows].float().mean().item())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(t.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            gstep += 1

            if d.main and gstep % log_every == 0:
                row = {k: sum(v) / len(v) for k, v in window.items()}
                log.log("train", gstep, lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm), **row)
                elapsed = time.monotonic() - t0
                logger.info("step {}/{} epoch {} loss {:.4f} lr {:.2e} | {:.1f} min", gstep, total_steps, epoch, row["loss"], scheduler.get_last_lr()[0], elapsed / 60)
                window.clear()

            now = time.monotonic()
            stop_steps = max_steps is not None and gstep >= max_steps
            stop_time = max_minutes is not None and (now - t0) / 60 >= max_minutes
            do_eval, do_ckpt, stop_steps, stop_time = d.broadcast_flags(
                gstep % eval_every == 0 or stop_steps, now - last_ckpt >= every_s or stop_time or stop_steps, stop_steps, stop_time,
                device=rt.device,
            )
            if do_eval and d.main:
                run_eval()
            if do_ckpt:
                if d.main:
                    checkpoint(epoch, s + 1)
                last_ckpt = time.monotonic()
            d.barrier()
            if stop_steps or stop_time:
                status = "max_steps" if stop_steps else "paused"
                break
        del it, loader
        if status != "done":
            break

    if status == "done" and d.main:
        logger.info("training done at step {}; final checkpoint and predictions", gstep)
        checkpoint(n_epochs - 1, steps_per_epoch[-1])
        predictor = Predictor(model, text_encoder, amp_dtype=rt.amp_dtype)
        preds = [p for split in EVAL_SPLITS if split in eval_sets for p in predictor.predict_records(eval_sets[split], eval_bs)]
        write_jsonl(run_dir / "preds.jsonl", preds)
        final = save_final(model, tokenizer, cfg, run_dir / "final")
        store.upload_folder(final, "final")
        store.upload_files(run_files() + [run_dir / "preds.jsonl"])
    if d.main:
        store.wait()
    d.barrier()
    logger.info("run {}: {} at step {} of {}", run_name, status, gstep, total_steps)
    return {"status": status, "step": gstep, "total_steps": total_steps, "run_dir": str(run_dir)}
