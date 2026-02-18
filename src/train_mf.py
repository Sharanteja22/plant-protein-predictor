import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR = "data/mf_tensors50"

X_train = torch.load(f"{DATA_DIR}/X_train.pt", weights_only=False)
y_train = torch.load(f"{DATA_DIR}/y_train.pt", weights_only=False)

X_val   = torch.load(f"{DATA_DIR}/X_val.pt", weights_only=False)
y_val   = torch.load(f"{DATA_DIR}/y_val.pt", weights_only=False)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)

batch_size = 32

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


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

model = MFClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

epochs = 30
patience = 5

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, epochs + 1):
    # ---- Train ----
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/mf_best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break


