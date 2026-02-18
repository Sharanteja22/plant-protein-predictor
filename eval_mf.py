import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
###phase 1 esm2 mf evaluation script after training best model saved as mf_best_model.pt##
# =========================
# Paths
# =========================
DATA_DIR = "data/mf_tensors50"
MODEL_PATH = "models/mf_best_model.pt"
BATCH_SIZE = 256

# =========================
# Load validation tensors
# =========================
print("Loading MF validation tensors...")

X_val = torch.load(f"{DATA_DIR}/X_val.pt", weights_only=False)
y_val = torch.load(f"{DATA_DIR}/y_val.pt", weights_only=False)

# 🔒 FORCE conversion to torch.Tensor (even if already tensor, this is safe)
X_val = torch.as_tensor(X_val, dtype=torch.float32)
y_val = torch.as_tensor(y_val, dtype=torch.float32)

print("Val tensors:", X_val.shape, y_val.shape)

# =========================
# Model definition (must match training)
# =========================
class MFClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_labels=323):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.BatchNorm1d(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Load model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MFClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# DataLoader (NOW SAFE)
# =========================
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# Collect probabilities
# =========================
all_probs = []
all_targets = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

print("Collected probabilities:", all_probs.shape)

# =========================
# Threshold sweep
# =========================
print("\n🔍 Threshold Sweep (Micro-F1)\n")

thresholds = np.arange(0.05, 0.41, 0.05)
best_f1 = 0.0
best_th = 0.0

for th in thresholds:
    preds = (all_probs >= th).astype(int)
    micro_f1 = f1_score(all_targets, preds, average="micro")

    print(f"Threshold {th:.2f} → Micro-F1: {micro_f1:.4f}")

    if micro_f1 > best_f1:
        best_f1 = micro_f1
        best_th = th

print(f"\n✅ Best Threshold: {best_th:.2f}")
print(f"✅ Best Micro-F1:  {best_f1:.4f}")

# =========================
# Top-K evaluation
# =========================
def topk_micro_f1(probs, targets, k):
    preds = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        topk_idx = np.argsort(probs[i])[-k:]
        preds[i, topk_idx] = 1
    return f1_score(targets, preds, average="micro")

print("\n🧪 Top-K Evaluation\n")

for k in [1, 3, 5]:
    score = topk_micro_f1(all_probs, all_targets, k)
    print(f"Top-{k} Micro-F1: {score:.4f}")
