# tree_train.py
# Training script for tree-LSTM similarity model
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from tree_model import TreePairClassifier, ContrastiveTreeModel
from tree_dataset import TreePairDataset, TreePairDataLoader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, opt, scaler, device, args):
    """Train for one epoch."""
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc="train", leave=False)
    for batch in progress_bar:
        # Move to device
        features1_list = [f.to(device) for f in batch["features1"]]
        children1_list = [c.to(device) for c in batch["children1"]]
        features2_list = [f.to(device) for f in batch["features2"]]
        children2_list = [c.to(device) for c in batch["children2"]]
        labels = batch["labels"].to(device)

        opt.zero_grad()

        with torch.amp.autocast('cuda', enabled=args.amp):
            # Forward pass
            predictions = model(
                (features1_list, children1_list),
                (features2_list, children2_list),
            )

            # Compute loss
            loss = F.binary_cross_entropy(predictions.squeeze(1), labels)

            # Add contrastive loss if using ContrastiveTreeModel
            if isinstance(model, ContrastiveTreeModel):
                # Normalize embeddings
                embeddings1 = model.projection_head(model.tree_encoder(features1_list, children1_list))
                embeddings2 = model.projection_head(model.tree_encoder(features2_list, children2_list))

                # Contrastive loss
                # Positive pairs (label=1): minimize distance
                # Negative pairs (label=0): maximize distance
                distances = 1 - F.cosine_similarity(embeddings1, embeddings2, dim=1)
                positive_mask = labels == 1
                negative_mask = labels == 0

                if positive_mask.sum() > 0:
                    positive_loss = distances[positive_mask].mean()
                else:
                    positive_loss = 0.0

                if negative_mask.sum() > 0:
                    negative_loss = F.relu(args.margin - distances[negative_mask]).mean()
                else:
                    negative_loss = 0.0

                contrastive_loss = positive_loss + args.contrastive_weight * negative_loss
                loss = loss + args.contrastive_lambda * contrastive_loss

        # Backward pass
        scaler.scale(loss).backward()

        if args.grad_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

        scaler.step(opt)
        scaler.update()

        losses.append(loss.item())
        preds = (predictions.squeeze(1) > 0.5).float().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc


@torch.no_grad()
def eval_epoch(model, loader, device, args):
    """Evaluate for one epoch."""
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="eval", leave=False):
        features1_list = [f.to(device) for f in batch["features1"]]
        children1_list = [c.to(device) for c in batch["children1"]]
        features2_list = [f.to(device) for f in batch["features2"]]
        children2_list = [c.to(device) for c in batch["children2"]]
        labels = batch["labels"].to(device)

        with autocast(enabled=args.amp):
            predictions = model(
                (features1_list, children1_list),
                (features2_list, children2_list),
            )
            loss = F.binary_cross_entropy(predictions.squeeze(1), labels)

        losses.append(loss.item())
        probs = predictions.squeeze(1).cpu().numpy()
        preds = (probs > 0.5).astype(float)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return np.mean(losses), acc, precision, recall, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train pairs JSON")
    parser.add_argument("--valid", type=str, default=None, help="Path to valid pairs JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="tree_out")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_type", type=str, default="classifier",
                       choices=["classifier", "contrastive"],
                       help="Model type: classifier or contrastive")
    parser.add_argument("--contrastive_weight", type=float, default=0.5,
                       help="Weight for negative pairs in contrastive loss")
    parser.add_argument("--contrastive_lambda", type=float, default=0.1,
                       help="Weight for contrastive loss overall")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for contrastive loss")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Determine feature dimension
    # Load a sample to check feature dim
    train_dataset = TreePairDataset.from_json(args.train)
    sample_item = train_dataset[0]
    feature_dim = sample_item["features1"].shape[1]
    print(f"Feature dimension: {feature_dim}")

    # Create model
    if args.model_type == "classifier":
        model = TreePairClassifier(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    else:
        model = ContrastiveTreeModel(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

    model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TreePairDataLoader.collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = None
    if args.valid:
        valid_dataset = TreePairDataset.from_json(args.valid)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=TreePairDataLoader.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    # Optimizer
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # Training loop
    best_valid_auc = 0.0
    best_epoch = 0

    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device: {args.device}")
    print()

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, opt, scaler, args.device, args)

        # Print progress
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        # Validate
        if valid_loader is not None:
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = eval_epoch(
                model, valid_loader, args.device, args
            )

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"          val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                      f"val_f1={val_f1:.4f} val_auc={val_auc:.4f}")

            # Save best model
            if val_auc > best_valid_auc:
                best_valid_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
                if epoch % 5 != 0 and epoch != args.epochs - 1:
                    print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
                    print(f"          val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                          f"val_f1={val_f1:.4f} val_auc={val_auc:.4f}")
                print(f"  -> Saved best model (AUC: {val_auc:.4f})")
        else:
            # Save each epoch if no validation
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

    print(f"\nTraining complete!")
    if valid_loader is not None:
        print(f"Best validation AUC: {best_valid_auc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
