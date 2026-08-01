# lightgbm_train.py
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import lightgbm as lgb
from dataset import NodeDataset
import joblib
from lightgbm import early_stopping, log_evaluation, callback


num_buckets = 5

def load_dataset_from_node_dataset(path):
    """
    Read a NodeDataset (assumes NodeDataset[path] yields dicts with "feat","bucket","log_q")
    Return: feats (N,D) numpy float32, buckets (N,) int, logq (N,) float32
    """
    ds = NodeDataset(path)
    feats_list = []
    buckets_list = []
    logq_list = []
    for i in tqdm(range(len(ds)), desc=f"loading {os.path.basename(path)}"):
        item = ds[i]
        feats_list.append(item["feat"].astype(np.float32))
        buckets_list.append(int(item["bucket"]))
        logq_list.append(float(item["log_q"]))
    feats = np.stack(feats_list, axis=0)
    buckets = np.array(buckets_list, dtype=np.int32)
    logq = np.array(logq_list, dtype=np.float32)
    return feats, buckets, logq

def compute_metrics_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    # lite group mapping: 0/1 -> 0 ; 2 -> 2 ; 3/4 -> 3
    group_map = {0:0, 1:0, 2:2, 3:3, 4:3}
    lite_true = [group_map[int(x)] for x in y_true]
    lite_pred = [group_map[int(x)] for x in y_pred]
    lite_acc = accuracy_score(lite_true, lite_pred)
    lite_recall = f1_score(lite_true, lite_pred, average="weighted", zero_division=0)
    return acc, f1, lite_acc, lite_recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="train dataset path (for NodeDataset)")
    parser.add_argument("--valid", type=str, default=None, help="valid dataset path (optional)")
    parser.add_argument("--out_dir", type=str, default="out", help="where to save models")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--verbose", type=int, default=-1)
    parser.add_argument("--save_preds", action="store_true", help="save predictions to out_dir/preds.npz")
    parser.add_argument("--train_classifier_only", action="store_true", help="only train classifier")
    parser.add_argument("--train_regressor_only", action="store_true", help="only train regressor")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("Loading datasets...")
    X_train, y_train_cls, y_train_reg = load_dataset_from_node_dataset(args.train)
    X_valid = y_valid_cls = y_valid_reg = None
    if args.valid:
        X_valid, y_valid_cls, y_valid_reg = load_dataset_from_node_dataset(args.valid)

    # LightGBM datasets
    lgb_train_cls = lgb.Dataset(X_train, label=y_train_cls, free_raw_data=False)
    lgb_train_reg = lgb.Dataset(X_train, label=y_train_reg, free_raw_data=False)

    valid_sets = []
    valid_names = []
    if args.valid:
        lgb_valid_cls = lgb.Dataset(X_valid, label=y_valid_cls, reference=lgb_train_cls, free_raw_data=False)
        lgb_valid_reg = lgb.Dataset(X_valid, label=y_valid_reg, reference=lgb_train_reg, free_raw_data=False)
        valid_sets = [lgb_valid_cls, lgb_valid_reg]
        valid_names = ["valid_cls", "valid_reg"]

    # common params
    common_params = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "verbose": args.verbose,
    }

    # 1) Multiclass classifier
    if not args.train_regressor_only:
        print("Training LightGBM classifier (multiclass)...")
        params_cls = {
            **common_params,
            "objective": "multiclass",
            "num_class": num_buckets,
            "metric": "multi_logloss",
        }
        evals_result_cls = {}
        # prepare container
        evals_result_cls = {}

        if args.valid:
            callbacks_cls = [
                callback.record_evaluation(evals_result_cls),
                early_stopping(stopping_rounds=args.early_stopping),
                log_evaluation(50)
            ]
            bst_cls = lgb.train(
                params_cls,
                lgb_train_cls,
                num_boost_round=args.num_rounds,
                valid_sets=[lgb_train_cls, lgb_valid_cls],
                valid_names=["train", "valid"],
                callbacks=callbacks_cls
            )
        else:
            # still record train metrics
            callbacks_cls = [callback.record_evaluation(evals_result_cls), log_evaluation(50)]
            bst_cls = lgb.train(
                params_cls,
                lgb_train_cls,
                num_boost_round=args.num_rounds,
                callbacks=callbacks_cls
            )
        # evals_result_cls will now contain training/validation metrics per iter (if any)


        cls_model_path = os.path.join(args.out_dir, "lgb_classifier.txt")
        bst_cls.save_model(cls_model_path)
        print(f"Saved classifier -> {cls_model_path}")
    else:
        bst_cls = None

    # 2) Regression for log_q
    if not args.train_classifier_only:
        print("Training LightGBM regressor (log_q)...")
        params_reg = {
            **common_params,
            "objective": "regression",
            "metric": "rmse",
        }
        evals_result_reg = {}

        if args.valid:
            callbacks_reg = [
                callback.record_evaluation(evals_result_reg),
                early_stopping(stopping_rounds=args.early_stopping),
                log_evaluation(50)
            ]
            bst_reg = lgb.train(
                params_reg,
                lgb_train_reg,
                num_boost_round=args.num_rounds,
                valid_sets=[lgb_train_reg, lgb_valid_reg],
                valid_names=["train", "valid"],
                callbacks=callbacks_reg
            )
        else:
            callbacks_reg = [callback.record_evaluation(evals_result_reg), log_evaluation(50)]
            bst_reg = lgb.train(
                params_reg,
                lgb_train_reg,
                num_boost_round=args.num_rounds,
                callbacks=callbacks_reg
            )

        reg_model_path = os.path.join(args.out_dir, "lgb_regressor.txt")
        bst_reg.save_model(reg_model_path)
        print(f"Saved regressor -> {reg_model_path}")
    else:
        bst_reg = None

    # Evaluation on validation or train (if valid not provided)
    eval_X = X_valid if args.valid else X_train
    eval_y_cls = y_valid_cls if args.valid else y_train_cls
    eval_y_reg = y_valid_reg if args.valid else y_train_reg

    results = {}
    if bst_cls is not None:
        print("Predicting with classifier...")
        probs = bst_cls.predict(eval_X, num_iteration=bst_cls.best_iteration or None)
        preds = np.argmax(probs, axis=1)
        acc, f1, lite_acc, lite_recall = compute_metrics_classification(eval_y_cls, preds)
        results.update({
            "cls_acc": acc,
            "cls_f1": f1,
            "cls_lite_acc": lite_acc,
            "cls_lite_recall": lite_recall
        })
        print(f"Classifier results - acc: {acc:.4f}, f1: {f1:.4f}, lite_acc: {lite_acc:.4f}, lite_recall: {lite_recall:.4f}")
    if bst_reg is not None:
        print("Predicting with regressor...")
        pred_reg = bst_reg.predict(eval_X, num_iteration=bst_reg.best_iteration or None)
        rmse = np.sqrt(mean_squared_error(eval_y_reg, pred_reg))
        mse = mean_squared_error(eval_y_reg, pred_reg)
        results.update({"reg_mse": mse, "reg_rmse": rmse})
        print(f"Regressor results - mse: {mse:.6f}, rmse: {rmse:.6f}")

    # If both models exist, optionally compute combined diagnostics (e.g., confusion or group metrics)
    if bst_cls is not None and bst_reg is not None:
        # Example: save predictions
        if args.save_preds:
            savep = os.path.join(args.out_dir, "preds.npz")
            np.savez(savep,
                     X=eval_X,
                     true_cls=eval_y_cls,
                     pred_cls=preds,
                     true_logq=eval_y_reg,
                     pred_logq=pred_reg)
            print(f"Saved predictions -> {savep}")

    # Save models as joblib as convenience (optional)
    try:
        if bst_cls is not None:
            joblib.dump(bst_cls, os.path.join(args.out_dir, "lgb_classifier.joblib"))
        if bst_reg is not None:
            joblib.dump(bst_reg, os.path.join(args.out_dir, "lgb_regressor.joblib"))
    except Exception as e:
        # joblib may not serialize Booster directly in some versions; ignore if fails
        print("Warning: joblib dump failed:", e)

    print("Training complete. Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
