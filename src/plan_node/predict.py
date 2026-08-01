# predict.py
import torch
import torch.nn.functional as F
import numpy as np
from plan_node.model import NodeEstimatorNet
from plan_node.utils import id2bucket

def _infer_input_dim_from_checkpoint(model_path, fallback_dim=None):
    state = torch.load(model_path, map_location="cpu")
    sd = state.get("state_dict", state)
    w = sd.get("fc1.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    if fallback_dim is not None:
        return int(fallback_dim)
    raise RuntimeError(f"Cannot infer input_dim from checkpoint: {model_path}")

def load_model(model_path, input_dim=None, hidden_dim=128, shared_dim=128, n_buckets=5, dropout=0.2, device='cuda'):
    if input_dim is None:
        input_dim = _infer_input_dim_from_checkpoint(model_path)

    model = NodeEstimatorNet(input_dim=input_dim, hidden_dim=hidden_dim, 
                           shared_dim=shared_dim, n_buckets=n_buckets, dropout=dropout)
    state = torch.load(model_path, map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model, feature_vector, device='cuda'):
    if isinstance(feature_vector, np.ndarray):
        feature_vector = torch.from_numpy(feature_vector).float()
    
    if feature_vector.dim() == 1:
        feature_vector = feature_vector.unsqueeze(0)
    feature_vector = feature_vector.to(device)
    

    with torch.no_grad():
        logits, pred_logq = model(feature_vector)
        pred_bucket_id = logits.argmax(dim=-1).item()
        pred_qerror = torch.exp(pred_logq).item()

    bucket_name = id2bucket[pred_bucket_id]
    
    return bucket_name, pred_qerror

def predict_from_vector(feature_vector, model_path = "models/cardinality_bias/best_model.pt", input_dim=None, hidden_dim=128, 
                       shared_dim=128, n_buckets=5, dropout=0.2, device='cuda'):
    
    model = load_model(model_path, input_dim, hidden_dim, shared_dim, n_buckets, dropout, device)
    return predict(model, feature_vector, device)


if __name__ == "__main__":
    example_vector = np.random.randn(128) 

    bucket_name, qerror = predict_from_vector(example_vector)
    
    print(f"预测桶: {bucket_name}")
    print(f"预测Q-Error: {qerror:.4f}")
