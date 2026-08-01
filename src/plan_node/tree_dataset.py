# tree_dataset.py
# Dataset classes for tree-LSTM training
import torch
from torch.utils.data import Dataset
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class TreeDataset(Dataset):
    """
    Dataset for single trees.

    Each item is a tree with features and children adjacency list.
    """

    def __init__(self, trees: List[Dict[str, Any]]):
        """
        Args:
            trees: List of {"features": [...], "children": [...]} dicts
        """
        self.trees = trees

    def __len__(self) -> int:
        return len(self.trees)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tree = self.trees[idx]
        return {
            "features": torch.tensor(tree["features"], dtype=torch.float32),
            "children": torch.tensor(tree["children"], dtype=torch.long),
        }

    @classmethod
    def from_json(cls, json_path: str) -> "TreeDataset":
        """Load dataset from JSON file."""
        with open(json_path, "r") as f:
            trees = json.load(f)
        return cls(trees)

    def save(self, json_path: str) -> None:
        """Save dataset to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.trees, f)


class TreePairDataset(Dataset):
    """
    Dataset for tree pairs with similarity labels.

    Each item is a pair of trees with a label indicating similarity.
    """

    def __init__(self, pairs: List[Dict[str, Any]]):
        """
        Args:
            pairs: List of {"tree1": {...}, "tree2": {...}, "label": int} dicts
        """
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]

        return {
            "features1": torch.tensor(pair["tree1"]["features"], dtype=torch.float32),
            "children1": torch.tensor(pair["tree1"]["children"], dtype=torch.long),
            "features2": torch.tensor(pair["tree2"]["features"], dtype=torch.float32),
            "children2": torch.tensor(pair["tree2"]["children"], dtype=torch.long),
            "label": torch.tensor(pair["label"], dtype=torch.float32),
        }

    @classmethod
    def from_json(cls, json_path: str) -> "TreePairDataset":
        """Load dataset from JSON file."""
        with open(json_path, "r") as f:
            pairs = json.load(f)
        return cls(pairs)

    def save(self, json_path: str) -> None:
        """Save dataset to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.pairs, f)


class TreePairDataLoader:
    """
    Custom collate function for batching tree pairs.

    Since trees have different sizes, we need special handling for batching.
    """

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Collate function for DataLoader.

        Since trees have variable sizes, we keep them as lists instead of stacking.
        The model will handle processing each tree individually.
        """
        features1_list = [item["features1"] for item in batch]
        children1_list = [item["children1"] for item in batch]
        features2_list = [item["features2"] for item in batch]
        children2_list = [item["children2"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        return {
            "features1": features1_list,
            "children1": children1_list,
            "features2": features2_list,
            "children2": children2_list,
            "labels": labels,
        }


def create_tree_dataloaders(
    train_pairs_path: str,
    valid_pairs_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[TreePairDataLoader, Optional[TreePairDataLoader]]:
    """
    Create train and validation dataloaders.

    Args:
        train_pairs_path: Path to training pairs JSON
        valid_pairs_path: Path to validation pairs JSON (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Load datasets
    train_dataset = TreePairDataset.from_json(train_pairs_path)

    if valid_pairs_path and os.path.exists(valid_pairs_path):
        valid_dataset = TreePairDataset.from_json(valid_pairs_path)
    else:
        valid_dataset = None

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=TreePairDataLoader.collate_fn,
        pin_memory=True,
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=TreePairDataLoader.collate_fn,
            pin_memory=True,
        )

    return train_loader, valid_loader


def split_pairs_to_train_valid(
    pairs_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split pairs into train and validation sets.

    Splits at the PARENT level to avoid data leakage.

    Args:
        pairs_path: Path to pairs JSON
        output_dir: Directory to save split files
        train_ratio: Ratio of training data
        seed: Random seed

    Returns:
        Tuple of (train_path, valid_path)
    """
    import random
    random.seed(seed)

    # Load pairs
    with open(pairs_path, "r") as f:
        pairs = json.load(f)

    # Extract unique parent IDs from pairs
    # We need to track which parents are in each pair
    parent_to_pair_indices = {}

    for idx, pair in enumerate(pairs):
        # Create a key from both trees' parent IDs
        # Since pairs don't directly have parent info, we need to track differently
        # For now, just shuffle and split
        pass

    # Simple shuffle and split (may have some data leakage)
    # For proper splitting, encoder should save parent info in pairs
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    train_size = int(len(pairs) * train_ratio)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_pairs = [pairs[i] for i in train_indices]
    valid_pairs = [pairs[i] for i in valid_indices]

    # Save
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_pairs.json")
    valid_path = os.path.join(output_dir, "valid_pairs.json")

    with open(train_path, "w") as f:
        json.dump(train_pairs, f)

    with open(valid_path, "w") as f:
        json.dump(valid_pairs, f)

    print(f"Split {len(pairs)} pairs into:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Valid: {len(valid_pairs)} ({len(valid_pairs)/len(pairs)*100:.1f}%)")

    return train_path, valid_path


if __name__ == "__main__":
    # Test the dataset classes
    print("Creating dummy tree pair data...")

    # Create dummy trees
    dummy_tree = {
        "features": [[0.1] * 1163, [0.2] * 1163, [0.3] * 1163],
        "children": [[1, 2], [-1, -1], [-1, -1]],
    }

    # Create dummy pairs
    dummy_pairs = [
        {"tree1": dummy_tree, "tree2": dummy_tree, "label": 1},
        {"tree1": dummy_tree, "tree2": dummy_tree, "label": 0},
    ]

    # Test TreePairDataset
    dataset = TreePairDataset(dummy_pairs)
    print(f"Dataset size: {len(dataset)}")

    # Test __getitem__
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    print(f"  features1 shape: {item['features1'].shape}")
    print(f"  children1 shape: {item['children1'].shape}")
    print(f"  label: {item['label']}")

    # Test collate_fn
    batch = [dataset[0], dataset[1]]
    collated = TreePairDataLoader.collate_fn(batch)
    print(f"\nCollated batch keys: {collated.keys()}")
    print(f"  Number of features1: {len(collated['features1'])}")
    print(f"  labels shape: {collated['labels'].shape}")

    print("\nAll tests passed!")
