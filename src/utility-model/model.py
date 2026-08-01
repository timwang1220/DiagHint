#!/usr/bin/env python3
"""v1 source-target utility scorer with shared TreeLSTM encoder."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class BinaryTreeLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_iou = nn.Linear(input_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(2 * hidden_dim, 3 * hidden_dim)
        self.W_f = nn.Linear(input_dim, 2 * hidden_dim)
        self.U_f = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                if "W_" in name:
                    nn.init.xavier_uniform_(param, gain=0.7)
                elif "U_" in name:
                    nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def _postorder(children: torch.Tensor) -> List[int]:
        n = int(children.shape[0])
        has_parent = [False] * n
        for i in range(n):
            l = int(children[i][0].item())
            r = int(children[i][1].item())
            if l != -1:
                has_parent[l] = True
            if r != -1:
                has_parent[r] = True

        roots = [i for i in range(n) if not has_parent[i]]
        seen = set()
        order: List[int] = []

        def dfs(x: int) -> None:
            if x == -1 or x in seen:
                return
            seen.add(x)
            l = int(children[x][0].item())
            r = int(children[x][1].item())
            dfs(l)
            dfs(r)
            order.append(x)

        for r in roots:
            dfs(r)
        return order

    def _forward_single(self, features: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
        n = int(features.shape[0])
        device = features.device
        zero = torch.zeros(1, self.hidden_dim, dtype=features.dtype, device=device)
        # Use per-node state lists to avoid in-place writes on a shared tensor
        # that can break autograd version tracking.
        h_states: List[torch.Tensor] = [zero for _ in range(n)]
        c_states: List[torch.Tensor] = [zero for _ in range(n)]
        order = self._postorder(children)

        for idx in order:
            x = features[idx].unsqueeze(0)
            left, right = children[idx].tolist()
            hl = h_states[left] if left != -1 else zero
            hr = h_states[right] if right != -1 else zero
            cl = c_states[left] if left != -1 else zero
            cr = c_states[right] if right != -1 else zero

            hc = torch.cat([hl, hr], dim=-1)
            iou = self.W_iou(x) + self.U_iou(hc)
            i, o, u = torch.chunk(iou, 3, dim=-1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)

            f = self.W_f(x) + self.U_f(hc)
            fl, fr = torch.chunk(torch.sigmoid(f), 2, dim=-1)
            c_v = i * u + fl * cl + fr * cr
            h_v = o * torch.tanh(c_v)
            c_states[idx] = c_v
            h_states[idx] = h_v

        root = order[-1]
        return h_states[root].squeeze(0)

    def forward(self, features_list: List[torch.Tensor], children_list: List[torch.Tensor]) -> torch.Tensor:
        out = [self._forward_single(f, c) for f, c in zip(features_list, children_list)]
        return torch.stack(out, dim=0)


class PairScorer(nn.Module):
    def __init__(self, tree_hidden_dim: int, mlp_hidden_dim: int, aux_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        in_dim = tree_hidden_dim * 4 + aux_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
        reuse: torch.Tensor,
        opt_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pair_feat = torch.cat(
            [
                h_source,
                h_target,
                torch.abs(h_source - h_target),
                h_source * h_target,
                reuse,
                opt_vec,
            ],
            dim=-1,
        )
        logit = self.mlp(pair_feat).squeeze(-1)
        score = torch.tanh(logit)
        return {"logit": logit, "score": score}


class SourceTargetTreeModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        tree_hidden_dim: int = 128,
        scorer_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = BinaryTreeLSTMEncoder(node_input_dim, tree_hidden_dim)
        self.scorer = PairScorer(tree_hidden_dim, scorer_hidden_dim, aux_dim=5, dropout=dropout)

    def encode_tree(self, features_list: List[torch.Tensor], children_list: List[torch.Tensor]) -> torch.Tensor:
        return self.encoder(features_list, children_list)

    def forward(
        self,
        source_features: List[torch.Tensor],
        source_children: List[torch.Tensor],
        target_features: List[torch.Tensor],
        target_children: List[torch.Tensor],
        reuse: torch.Tensor,
        opt_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_source = self.encode_tree(source_features, source_children)
        h_target = self.encode_tree(target_features, target_children)
        return self.scorer(h_source, h_target, reuse, opt_vec)
