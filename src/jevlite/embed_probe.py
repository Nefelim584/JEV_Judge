"""Linear probe on a frozen embedding model (Qwen3-Embedding): is the backbone the limit?

A cheap stand-in for fine-tuning a large backbone. The frozen model embeds ``statement + text`` in its own
instruction format; a logistic regression on the embeddings, trained on Bool records of a training mix, then
scores the criteria bench items. If a frozen 8B + one linear layer already matches the fine-tuned jev-lite on
the sources jev-lite never saw, the backbone (not the data) is the limit; if it falls far short, a fine-tuned
8B is unlikely to be worth its price. A probe understates a fine-tune, so only the first outcome is conclusive.

    ids = query_ids(tokenizer, prompt, state, max_len=1024, readout_id=eot)   # Instruct: … Query:… <|endoftext|>
    x = embed(model, rows, pad_id, "cuda")                                     # [N, H] fp16, L2-normalised
    clf, info = fit_probe(x_train, y_train)                                    # C picked on a held-out 20%
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np

from .encoding import serialize_state, truncate_head_tail

INSTRUCT = "Decide whether the statement is true of the text. Statement: {prompt}"
READOUT_TOKEN = "<|endoftext|>"


def query_ids(tokenizer, prompt: str, state, max_len: int, readout_id: int) -> list[int]:
    """``Instruct: <statement>\\nQuery:<text><|endoftext|>`` (Qwen3-Embedding's query format), at most
    ``max_len`` tokens. The text is cut head and tail; the statement only if it alone takes over half."""
    def tok(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    prefix, _ = truncate_head_tail(tok(f"Instruct: {INSTRUCT.format(prompt=prompt)}\nQuery:"), max_len // 2, 0.5)
    body, _ = truncate_head_tail(tok(serialize_state(state)), max_len - len(prefix) - 1, 0.5)
    return [*prefix, *body, readout_id]


def hard_label(target: float, margin: float = 0.2) -> int | None:
    """0 / 1 for a confident target, ``None`` for a soft one in between (left out of the probe)."""
    if target <= margin:
        return 0
    if target >= 1 - margin:
        return 1
    return None


def sample_bool_records(rows: Iterable[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to ``n`` Bool records with a confident target, drawn at random (seeded)."""
    pool = [r for r in rows if r.get("type") == "bool" and hard_label(float(r["target"])) is not None]
    rng = random.Random(seed)
    return rng.sample(pool, n) if n < len(pool) else pool


def token_batches(lengths: Sequence[int], tokens_per_batch: int) -> list[list[int]]:
    """Indices grouped longest first, so that ``rows × longest row`` stays within ``tokens_per_batch``
    (a single row longer than the budget gets a batch of its own)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    for i in order:
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= tokens_per_batch:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def embed(model, rows: Sequence[list[int]], pad_id: int, device, tokens_per_batch: int = 32768) -> np.ndarray:
    """``[N, H]`` fp16 embeddings in input order: the hidden state of each row's last token, L2-normalised.
    Rows are right-padded, so the last real token never attends to padding (causal attention)."""
    import torch
    import torch.nn.functional as F

    out = np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for batch in token_batches([len(r) for r in rows], tokens_per_batch):
            length = max(len(rows[i]) for i in batch)
            ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), length), dtype=torch.long)
            for j, i in enumerate(batch):
                ids[j, : len(rows[i])] = torch.tensor(rows[i])
                mask[j, : len(rows[i])] = 1
            hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            last = hidden[torch.arange(len(batch), device=hidden.device), mask.sum(1).to(hidden.device) - 1]
            out[batch] = F.normalize(last.float(), dim=-1).cpu().numpy().astype(np.float16)
    return out


def fit_probe(x: np.ndarray, y: np.ndarray, seed: int = 0, cs: Sequence[float] = (0.1, 1.0, 10.0, 100.0)):
    """Logistic regression; ``C`` picked by AUROC on a stratified 20% of the training records, then refit
    on all of them. Returns ``(classifier, info)``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    x = np.asarray(x, dtype=np.float32)
    x_fit, x_val, y_fit, y_val = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    val = {}
    for c in cs:
        clf = LogisticRegression(C=c, max_iter=3000).fit(x_fit, y_fit)
        val[c] = float(roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1]))
    best = max(val, key=val.get)
    clf = LogisticRegression(C=best, max_iter=3000).fit(x, y)
    return clf, {"C": best, "val_auroc": {str(c): v for c, v in val.items()}, "n_train": int(len(y)),
                 "positive_rate": float(np.mean(y))}
