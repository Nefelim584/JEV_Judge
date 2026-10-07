"""Phase 5 sanity check: a random-init model must overfit 100 training records to near-zero loss.

    uv run python scripts/overfit_sanity.py                                   # small random ModernBERT, ~1–3 min
    uv run python scripts/overfit_sanity.py --size large --steps 400          # ModernBERT-large config, random init

Uses the real pipeline end to end: unified records → TrainCollator (Choice options shuffled per epoch)
→ JevLite → jev_loss. Passes if the final loss over all records (eval mode, options in the stored
order) is below ``--threshold`` and every record is predicted correctly. Exits 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from jevlite.collate import TYPE_IDS
from jevlite.data.loader import JsonlRecords, TrainCollator
from jevlite.encoding import PackedEncoder
from jevlite.losses import LossConfig, jev_loss
from jevlite.model import JevLite, question_probs

SIZES = {
    "small": dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4),
    "large": dict(hidden_size=1024, intermediate_size=2624, num_hidden_layers=28, num_attention_heads=16),
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def correct(grouped: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
    probs = question_probs(grouped, qtype)
    is_bool = qtype == TYPE_IDS["bool"]
    multi = probs.argmax(-1) == target.argmax(-1)
    return torch.where(is_bool, (probs[:, 0] > 0.5) == (target[:, 0] > 0.5), multi)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/mix/stage_a/train.e0.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--no-shuffle", action="store_true", help="keep the stored option order in training")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    records = JsonlRecords(args.data)
    records.lines = records.lines[: args.n]
    data = [records[i] for i in range(len(records))]
    by_type = {t: sum(r.type == t for r in data) for t in TYPE_IDS}
    print(f"{len(data)} records from {args.data}: {by_type}; device {device}, size {args.size}", flush=True)

    encoder = ModernBertModel(ModernBertConfig(
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, reference_compile=False,
        attn_implementation="sdpa", **SIZES[args.size],
    ))
    model = JevLite(encoder, hidden=SIZES[args.size]["hidden_size"], trunk_dim=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30))
    text_encoder = PackedEncoder(tokenizer, args.max_len)
    train_collate = TrainCollator(text_encoder, seed=args.seed, shuffle=not args.no_shuffle)
    eval_collate = TrainCollator(text_encoder, shuffle=False)
    loss_cfg = LossConfig()

    gen = torch.Generator().manual_seed(args.seed)
    step, epoch, start = 0, 0, time.perf_counter()
    model.train()
    while step < args.steps:
        train_collate.set_epoch(epoch)
        order = torch.randperm(len(data), generator=gen).tolist()
        for i in range(0, len(order), args.batch_size):
            batch, target = train_collate([data[j] for j in order[i : i + args.batch_size]])
            batch, target = batch.to(device), target.to(device)
            loss, parts = jev_loss(model.grouped_logits(batch, apply_temperature=False), target, batch, loss_cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                p = " ".join(f"{k} {v.item():.3f}" for k, v in parts.items())
                print(f"step {step:4d}  epoch {epoch:3d}  loss {loss.item():.4f}  ({p})  {time.perf_counter() - start:.0f}s", flush=True)
            if step >= args.steps:
                break
        epoch += 1

    model.eval()
    total, n, n_correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch, target = eval_collate(data[i : i + args.batch_size])
            batch, target = batch.to(device), target.to(device)
            grouped = model.grouped_logits(batch, apply_temperature=False)
            loss, _ = jev_loss(grouped, target, batch, loss_cfg)
            total += loss.item() * batch.n_questions
            n += batch.n_questions
            n_correct += int(correct(grouped, target, batch.question_type).sum())
    final = total / n
    ok = final < args.threshold and n_correct == n
    print(f"final loss {final:.4f} (threshold {args.threshold}), accuracy {n_correct}/{n} → {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
