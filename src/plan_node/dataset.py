# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class NodeDataset(Dataset):
    def __init__(self, data_dir):
        # load precomputed X, y, qerrors
        self.X = np.load(os.path.join(data_dir, "X.npy"))
        self.y = np.load(os.path.join(data_dir, "y.npy"))
        self.q = np.load(os.path.join(data_dir, "qerrors.npy"))
        # optional: alias info
        alias_path = os.path.join(data_dir, "alias2id.json")
        self.alias2id = json.load(open(alias_path)) if os.path.exists(alias_path) else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "feat": self.X[idx],
            "bucket": self.y[idx],
            "log_q": np.log1p(self.q[idx])
        }
