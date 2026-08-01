# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeEstimatorNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, shared_dim=256, n_buckets=5, dropout=0.2):
        super().__init__()
        # input_dim should come from encoded feature artifacts/config at train/infer time.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, shared_dim)
        self.ln2 = nn.LayerNorm(shared_dim)

        # classification head
        self.class_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim//2, n_buckets)
        )
        # regression head (log qerror)
        self.reg_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim//2, 1)
        )

    def forward(self, x):
        # x: [B, D]
        x = self.fc1(x)
        x = F.gelu(self.ln1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.gelu(self.ln2(x))

        cls_logits = self.class_head(x)
        reg_val = self.reg_head(x).squeeze(-1)
        return cls_logits, reg_val
