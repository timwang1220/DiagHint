# train.py
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from dataset import NodeDataset
from model import *      
from utils import bucket2id, id2bucket
import torch.nn.functional as F


num_buckets = 5


def collate_fn(batch):
    feats = np.stack([b["feat"] for b in batch]).astype(np.float32)
    feats = torch.from_numpy(feats)

    buckets = torch.tensor([b["bucket"] for b in batch], dtype=torch.long)
    logq = torch.tensor([b["log_q"] for b in batch], dtype=torch.float32)
    return feats, buckets, logq


def train_epoch(model, loader, opt, scaler, device, args):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    bucket_to_metaclass = torch.tensor([0, 0, 1, 2, 2], device=args.device)
    for step, (feats, buckets, logq) in enumerate(tqdm(loader, desc="train", position=0, leave=False)):
        feats = feats.to(device)
        buckets = buckets.to(device)
        logq = logq.to(device)

        with torch.amp.autocast('cuda', enabled=args.amp):
            logits, pred_logq = model(feats)
            c_loss = F.cross_entropy(logits, buckets)
            r_loss = F.mse_loss(pred_logq, logq)
            probs = F.softmax(logits, dim=1)
            true_metaclasses = bucket_to_metaclass[buckets]            
            all_metaclasses = bucket_to_metaclass.unsqueeze(0).expand(logits.size(0), -1)
            true_metaclasses_expanded = true_metaclasses.unsqueeze(1).expand(-1, num_buckets)
            penalty_mask = (all_metaclasses != true_metaclasses_expanded).float()
            wrong_metaclass_probs = probs * penalty_mask
            penalty_per_sample = torch.sum(wrong_metaclass_probs, dim=1)
            penalty_loss = penalty_per_sample.mean()
            loss = c_loss + args.lambda_reg * r_loss + args.gamma * penalty_loss


        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        losses.append(loss.item())
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(buckets.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return np.mean(losses), acc, f1

def compute_directional_lite_accuracy(labels, preds, qerrors):
    """
    Improved Lite accuracy: considers prediction direction.
    - If predicting 0/1 (underestimation) but label is 2: correct if qerror < 1
    - If predicting 3/4 (overestimation) but label is 2: correct if qerror > 1
    """
    correct = 0
    for label, pred, q in zip(labels, preds, qerrors):
        if label == pred:
            correct += 1
        elif label == 2:  # True label is Approximately Accurate
            if pred in [0, 1] and q < 1:  # Predicted underestimation and actually underestimated
                correct += 1
            elif pred in [3, 4] and q > 1:  # Predicted overestimation and actually overestimated
                correct += 1
    return correct / len(labels) if len(labels) > 0 else 0.0


@torch.no_grad()
def eval_epoch(model, loader, device, args):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    all_qerrors = []
    for feats, buckets, logq in tqdm(loader, desc="eval", position=1, leave=False):
        feats = feats.to(device)
        buckets = buckets.to(device)
        logq = logq.to(device)
        logits, pred_logq = model(feats)
        c_loss = F.cross_entropy(logits, buckets)
        r_loss = F.mse_loss(pred_logq, logq)
        loss = c_loss + args.lambda_reg * r_loss
        losses.append(loss.item())
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(buckets.detach().cpu().numpy().tolist())
        # Convert logq back to q-error
        qerrors = torch.expm1(logq).detach().cpu().numpy()
        all_qerrors.extend(qerrors.tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    # 计算方向感知的lite准确率：考虑预测方向是否正确
    lite_acc = compute_directional_lite_accuracy(all_labels, all_preds, all_qerrors)
    lite_recall = lite_acc  # Same metric for both
    return np.mean(losses), acc, f1, lite_acc, lite_recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--valid", type=str, required=False, default=None)
    parser.add_argument("--emb_dir", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dataset
    train_ds = NodeDataset(args.train)  # args.train 直接指向 artifacts 文件夹
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = None
    if args.valid:
        valid_ds = NodeDataset(args.valid)  # args.valid 直接指向 artifacts 文件夹
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    # infer input dim from dataset
    sample = train_ds[0]["feat"]
    input_dim = sample.shape[0]
    print(f"Input dimension: {input_dim}")
    model = NodeEstimatorNet(input_dim=input_dim, hidden_dim=128, shared_dim=128, n_buckets=num_buckets, dropout=0.2)
    model.to(args.device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    best_val = -1
    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, opt, scaler, args.device, args)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")

        if valid_loader:
            val_loss, val_acc, val_f1, val_lite_acc, val_lite_recall = eval_epoch(model, valid_loader, args.device, args)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_lite_acc={val_lite_acc:.4f} val_lite_recall={val_lite_recall:.4f}")
            # save best
            if val_f1 > best_val:
                best_val = val_f1
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
                if epoch % 10 != 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")
                    print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_lite_acc={val_lite_acc:.4f} val_lite_recall={val_lite_recall:.4f}")
                print("Saved best model.")
        else:
            # save each epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

if __name__ == "__main__":
    main()
