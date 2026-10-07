import math

import pytest
import torch
import torch.nn.functional as F

from jevlite.collate import TYPE_IDS, Batch
from jevlite.losses import LossConfig, emd2, jev_loss, loss_targets, soft_cross_entropy

NEG_INF = float("-inf")


def _batch(types: list[str], n_candidates: list[int]) -> Batch:
    """A batch with only the fields the losses read."""
    k_max = max(n_candidates)
    mask = torch.arange(k_max)[None, :] < torch.tensor(n_candidates)[:, None]
    empty = torch.zeros(0, dtype=torch.long)
    return Batch(
        input_ids=empty, attention_mask=empty, marker_row=empty, marker_pos=empty, marker_type=empty,
        marker_question=empty, marker_candidate=empty,
        question_type=torch.tensor([TYPE_IDS[t] for t in types]), candidate_mask=mask,
        truncated=torch.zeros(len(types), dtype=torch.bool), question_ids=[f"q{i}" for i in range(len(types))],
    )


def test_emd_is_zero_for_identical_distributions():
    mask = torch.ones(3, 4, dtype=torch.bool)
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, 0, 0, 0], [0.25] * 4])
    assert torch.allclose(emd2(p, p, mask), torch.zeros(3))


def test_emd_punishes_far_levels_more():
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = torch.tensor([[0.0, 0, 0, 1]])  # critical
    near = emd2(torch.tensor([[0.0, 0, 1, 0]]), target, mask)  # high
    far = emd2(torch.tensor([[1.0, 0, 0, 0]]), target, mask)  # low
    assert far.item() == pytest.approx(1.0) and near.item() == pytest.approx(1 / 3)


def test_emd_ignores_padding():
    p, t = torch.tensor([[0.2, 0.8, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    two = emd2(p[:, :2], t[:, :2], torch.ones(1, 2, dtype=torch.bool))
    padded = emd2(p, t, torch.tensor([[True, True, False]]))
    assert padded.item() == pytest.approx(two.item()) == pytest.approx(0.64)


def test_soft_cross_entropy_matches_torch_and_ignores_padding():
    logits = torch.tensor([[1.0, 2.0, 0.5, NEG_INF], [0.3, -1.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, True, False], [True] * 4])
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    ce = soft_cross_entropy(logits, target, mask)
    assert torch.isfinite(ce).all()
    assert ce[0].item() == pytest.approx(F.cross_entropy(logits[:1, :3], torch.tensor([1])).item())
    assert ce[1].item() == pytest.approx(F.cross_entropy(logits[1:], torch.tensor([3])).item())


def test_loss_targets_hard_soft_and_bool():
    batch = _batch(["choice", "score", "bool"], [3, 4, 1])
    t = loss_targets([2, [0.1, 0.2, 0.3, 0.4], 0.7], batch)
    assert t.shape == (3, 4)
    assert t[0].tolist() == [0, 0, 1, 0]
    assert t[1].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert t[2].tolist() == pytest.approx([0.7, 0, 0, 0])
    with pytest.raises(ValueError, match="out of range"):
        loss_targets([3, 0, 1.0], batch)
    with pytest.raises(ValueError, match="bool target"):
        loss_targets([0, 0, 1.5], batch)


def test_jev_loss_parts_and_weights():
    batch = _batch(["choice", "score", "bool", "bool"], [3, 3, 1, 1])
    grouped = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.5, NEG_INF, NEG_INF], [-0.5, NEG_INF, NEG_INF]])
    target = loss_targets([0, 2, 1.0, 0.0], batch)
    total, parts = jev_loss(grouped, target, batch, LossConfig(emd_lambda=0.0, brier_mu=0.0))
    assert set(parts) == {"choice", "score", "bool"}
    assert parts["choice"].item() == pytest.approx(F.cross_entropy(grouped[:1], torch.tensor([0])).item())
    bce = F.binary_cross_entropy_with_logits(torch.tensor([1.5, -0.5]), torch.tensor([1.0, 0.0]))
    assert parts["bool"].item() == pytest.approx(bce.item())
    assert total.item() == pytest.approx(sum(p.item() for p in parts.values()))

    weighted, _ = jev_loss(grouped, target, batch, LossConfig(weights={"choice": 2.0, "score": 0.0, "bool": 1.0}, emd_lambda=0.0, brier_mu=0.0))
    assert weighted.item() == pytest.approx(2 * parts["choice"].item() + parts["bool"].item())


def test_jev_loss_extra_terms_and_gradients():
    batch = _batch(["score", "bool"], [4, 1])
    grouped = torch.tensor([[0.5, 0.1, -0.3, 0.2], [0.8, NEG_INF, NEG_INF, NEG_INF]], requires_grad=True)
    target = loss_targets([3, 0.0], batch)
    plain, plain_parts = jev_loss(grouped, target, batch, LossConfig(emd_lambda=0.0, brier_mu=0.0))
    full, parts = jev_loss(grouped, target, batch, LossConfig(emd_lambda=0.5, brier_mu=0.1))
    probs = F.softmax(grouped[:1].detach(), -1)
    assert parts["score"].item() == pytest.approx(plain_parts["score"].item() + 0.5 * emd2(probs, target[:1], batch.candidate_mask[:1]).item())
    assert parts["bool"].item() == pytest.approx(plain_parts["bool"].item() + 0.1 * torch.sigmoid(torch.tensor(0.8)).item() ** 2)
    full.backward()
    assert torch.isfinite(grouped.grad).all()
    assert grouped.grad[1, 1:].abs().sum() == 0  # padding gets no gradient


def test_perfect_prediction_has_near_zero_loss():
    batch = _batch(["choice", "score", "bool"], [3, 3, 1])
    big = 30.0
    grouped = torch.tensor([[big, 0, 0], [0, 0, big], [big, NEG_INF, NEG_INF]])
    total, _ = jev_loss(grouped, loss_targets([0, 2, 1.0], batch), batch)
    assert total.item() < 1e-6 and not math.isnan(total.item())
