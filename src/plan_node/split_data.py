# split_data.py
# Script to split encoded data into train and validation sets
# Splits by PLAN FILE to avoid data leakage
import os
import sys
import argparse
import numpy as np
import random
import shutil
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_data_by_plan(input_dir, train_dir, valid_dir, train_ratio=0.8, seed=42):
    """
    Split encoded data into train and validation sets BY PLAN FILE.
    This ensures no data leakage - all nodes from the same plan go to the same split.

    Args:
        input_dir: Directory containing X.npy, y.npy, qerrors.npy, plan_file_indices.npy
        train_dir: Directory to save training data
        valid_dir: Directory to save validation data
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed for reproducibility
    """
    # Load data
    print(f"Loading data from {input_dir}...")
    X = np.load(os.path.join(input_dir, "X.npy"))
    y = np.load(os.path.join(input_dir, "y.npy"))
    qerrors = np.load(os.path.join(input_dir, "qerrors.npy"))

    # Load plan file indices
    plan_file_indices_path = os.path.join(input_dir, "plan_file_indices.npy")
    if os.path.exists(plan_file_indices_path):
        plan_file_indices = np.load(plan_file_indices_path)
        print(f"Found plan file indices - splitting by plan to avoid leakage")
        split_by_plan = True
    else:
        print(f"WARNING: plan_file_indices.npy not found - falling back to node-level split")
        print(f"         This may cause data leakage!")
        split_by_plan = False

    n_samples = len(X)
    n_plan_files = len(np.unique(plan_file_indices)) if split_by_plan else 1
    print(f"Total samples: {n_samples}")
    print(f"Feature dimension: {X.shape[1]}")
    if split_by_plan:
        print(f"Number of plan files: {n_plan_files}")

    # Set random seed
    rng = np.random.RandomState(seed)

    if split_by_plan:
        # Group nodes by plan file
        unique_plan_ids = np.unique(plan_file_indices)
        n_plans = len(unique_plan_ids)

        # Shuffle plan IDs
        shuffled_plan_ids = rng.permutation(n_plans)

        # Split plans into train/valid
        train_size = int(n_plans * train_ratio)
        train_plan_ids = set(shuffled_plan_ids[:train_size])
        valid_plan_ids = set(shuffled_plan_ids[train_size:])

        # Create masks for samples
        train_mask = np.isin(plan_file_indices, list(train_plan_ids))
        valid_mask = np.isin(plan_file_indices, list(valid_plan_ids))

        train_indices = np.where(train_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        print(f"\nSplit by plan file:")
        print(f"  Train plans: {len(train_plan_ids)} ({len(train_plan_ids)/n_plans*100:.1f}%)")
        print(f"  Valid plans: {len(valid_plan_ids)} ({len(valid_plan_ids)/n_plans*100:.1f}%)")
    else:
        # Fallback: node-level split (may cause leakage)
        indices = rng.permutation(n_samples)
        train_size = int(n_samples * train_ratio)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

    print(f"\nResult:")
    print(f"  Train samples: {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"  Valid samples: {len(valid_indices)} ({len(valid_indices)/n_samples*100:.1f}%)")

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Split and save training data
    print(f"\nSaving training data to {train_dir}...")
    np.save(os.path.join(train_dir, "X.npy"), X[train_indices])
    np.save(os.path.join(train_dir, "y.npy"), y[train_indices])
    np.save(os.path.join(train_dir, "qerrors.npy"), qerrors[train_indices])
    # Also save plan file indices for verification
    if split_by_plan:
        np.save(os.path.join(train_dir, "plan_file_indices.npy"), plan_file_indices[train_indices])

    # Copy config and norm_stats if they exist
    for file in ["config.json", "norm_stats.npy", "plan_file_names.json"]:
        src = os.path.join(input_dir, file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(train_dir, file))

    # Split and save validation data
    print(f"Saving validation data to {valid_dir}...")
    np.save(os.path.join(valid_dir, "X.npy"), X[valid_indices])
    np.save(os.path.join(valid_dir, "y.npy"), y[valid_indices])
    np.save(os.path.join(valid_dir, "qerrors.npy"), qerrors[valid_indices])
    if split_by_plan:
        np.save(os.path.join(valid_dir, "plan_file_indices.npy"), plan_file_indices[valid_indices])

    # Copy config and norm_stats if they exist
    for file in ["config.json", "norm_stats.npy", "plan_file_names.json"]:
        src = os.path.join(input_dir, file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(valid_dir, file))

    print("\nDone!")
    print(f"Train set: {train_dir}")
    print(f"Valid set: {valid_dir}")

    # Verify no overlap in plan files
    if split_by_plan:
        train_plans = set(plan_file_indices[train_indices])
        valid_plans = set(plan_file_indices[valid_indices])
        overlap = train_plans & valid_plans
        if overlap:
            print(f"\nWARNING: Found {len(overlap)} overlapping plan files!")
        else:
            print(f"\nVerified: No overlapping plan files between train and valid")

    # Print bucket distribution
    print("\nBucket distribution:")
    print(f"  Train: {np.bincount(y[train_indices])}")
    print(f"  Valid: {np.bincount(y[valid_indices])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split encoded data into train/valid sets by plan file")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with X.npy, y.npy, qerrors.npy, plan_file_indices.npy")
    parser.add_argument("--train_dir", type=str, default=None, help="Output directory for training data")
    parser.add_argument("--valid_dir", type=str, default=None, help="Output directory for validation data")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    args = parser.parse_args()

    # Default output directories
    if args.train_dir is None:
        args.train_dir = os.path.join(os.path.dirname(args.input_dir), "train_artifacts")
    if args.valid_dir is None:
        args.valid_dir = os.path.join(os.path.dirname(args.input_dir), "valid_artifacts")

    split_data_by_plan(args.input_dir, args.train_dir, args.valid_dir, args.train_ratio, args.seed)
