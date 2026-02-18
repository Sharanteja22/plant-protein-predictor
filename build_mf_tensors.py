import pandas as pd
import ast
import torch
import numpy as np
##step 5: Building MF tensors for train, val, test splits Phase 1 After conversion of esm2 embeddings pool##
# -------------------------------
# PATHS
# -------------------------------
EMB_PATH = "data/esm2_embeddings.pt"
LABEL_PATH = "data/dataset_go_filtered.csv"

TRAIN_PATH = "data/train50.csv"
VAL_PATH   = "data/val50.csv"
TEST_PATH  = "data/test50.csv"

OUT_DIR = "data/mf_tensors50"
import os

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading embeddings...")
embeddings = torch.load(EMB_PATH)   # dict: protein_id -> (1280,)

print("Loading GO labels...")
df_labels = pd.read_csv(LABEL_PATH)
df_labels["go_MF"] = df_labels["go_MF"].apply(ast.literal_eval)

print("Loading splits...")
df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
df_test  = pd.read_csv(TEST_PATH)

# -------------------------------
# BUILD MF LABEL INDEX
# -------------------------------
print("Building MF label index...")
all_mf_terms = sorted(
    {go for gos in df_labels["go_MF"] for go in gos}
)

go_to_idx = {go: i for i, go in enumerate(all_mf_terms)}
num_labels = len(go_to_idx)

print("Total MF labels:", num_labels)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def build_X_y(df_split):
    X = []
    Y = []

    for _, row in df_split.iterrows():
        pid = row["protein_id"]

        if pid not in embeddings:
            continue

        # ---------- X ----------
        X.append(embeddings[pid])

        # ---------- y ----------
        gos = df_labels.loc[
            df_labels["protein_id"] == pid, "go_MF"
        ].values[0]

        y = np.zeros(num_labels, dtype=np.float32)
        for go in gos:
            y[go_to_idx[go]] = 1.0

        Y.append(y)

    return np.stack(X), np.stack(Y)

# -------------------------------
# BUILD TENSORS
# -------------------------------
print("Building train tensors...")
X_train, y_train = build_X_y(df_train)

print("Building val tensors...")
X_val, y_val = build_X_y(df_val)

print("Building test tensors...")
X_test, y_test = build_X_y(df_test)

# -------------------------------
# SAVE
# -------------------------------
torch.save(X_train, f"{OUT_DIR}/X_train.pt")
torch.save(y_train, f"{OUT_DIR}/y_train.pt")

torch.save(X_val, f"{OUT_DIR}/X_val.pt")
torch.save(y_val, f"{OUT_DIR}/y_val.pt")

torch.save(X_test, f"{OUT_DIR}/X_test.pt")
torch.save(y_test, f"{OUT_DIR}/y_test.pt")

print("\n✅ STEP 5 COMPLETE")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)
