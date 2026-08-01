#!/usr/bin/env python3
"""Pairwise-ranking + sign-consistency losses for utility-model v1."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_item = F.smooth_l1_loss(pred, target, reduction="none")
    return (per_item * weight).sum() / (weight.sum() + 1e-8)


def pairwise_logistic_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_ids: List[str],
    item_weight: torch.Tensor,
    min_delta: float = 0.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, tid in enumerate(target_ids):
        groups[tid].append(i)

    pair_losses = []
    pair_weights = []
    used_pairs = 0

    temp = max(float(temperature), 1e-6)
    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for ai in range(len(idxs)):
            i = idxs[ai]
            for bi in range(ai + 1, len(idxs)):
                j = idxs[bi]
                yi = labels[i]
                yj = labels[j]
                dy = yi - yj
                if torch.abs(dy).item() <= float(min_delta):
                    continue

                sign = torch.sign(dy)  # +1 means i should rank above j
                score_diff = (scores[i] - scores[j]) / temp
                # logistic ranking loss: log(1 + exp(-sign * (si - sj)))
                loss_ij = F.softplus(-sign * score_diff)
                weight_ij = 0.5 * (item_weight[i] + item_weight[j])

                pair_losses.append(loss_ij)
                pair_weights.append(weight_ij)
                used_pairs += 1

    if not pair_losses:
        return scores.new_tensor(0.0), 0
    ls = torch.stack(pair_losses)
    ws = torch.stack(pair_weights)
    return (ls * ws).sum() / (ws.sum() + 1e-8), used_pairs


def sign_consistency_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 0.01,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable sign/band consistency loss with tolerance band [-eps, +eps]."""
    e = float(eps)
    temp = max(float(temperature), 1e-6)

    pos_mask = labels > e
    neg_mask = labels < -e
    mid_mask = ~(pos_mask | neg_mask)

    per_item = torch.zeros_like(scores)

    # y > eps => encourage score > +eps
    if torch.any(pos_mask):
        per_item[pos_mask] = F.softplus(-((scores[pos_mask] - e) / temp))
    # y < -eps => encourage score < -eps
    if torch.any(neg_mask):
        per_item[neg_mask] = F.softplus(((scores[neg_mask] + e) / temp))
    # |y| <= eps => encourage score in [-eps, +eps]
    if torch.any(mid_mask):
        excess = (torch.abs(scores[mid_mask]) - e) / temp
        per_item[mid_mask] = F.softplus(excess)

    return (per_item * weight).sum() / (weight.sum() + 1e-8)


def total_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    target_ids: List[str],
    lambda_pair: float = 0.45,
    lambda_sign: float = 0.45,
    lambda_reg: float = 0.1,
    rank_min_delta: float = 0.0,
    rank_temperature: float = 1.0,
    sign_eps: float = 0.01,
    sign_temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pair, pair_count = pairwise_logistic_ranking_loss(
        scores=scores,
        labels=labels,
        target_ids=target_ids,
        item_weight=weight,
        min_delta=rank_min_delta,
        temperature=rank_temperature,
    )
    sign = sign_consistency_loss(
        scores=scores,
        labels=labels,
        weight=weight,
        eps=sign_eps,
        temperature=sign_temperature,
    )
    reg = weighted_huber_loss(scores, labels, weight)
    total = lambda_pair * pair + lambda_sign * sign + lambda_reg * reg
    return {
        "total": total,
        "pair": pair,
        "sign": sign,
        "reg": reg,
        "pair_count": logits.new_tensor(float(pair_count)),
    }
