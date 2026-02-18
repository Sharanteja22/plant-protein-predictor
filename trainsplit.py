import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = "data/dataset_go_filtered.csv"
TRAIN_PATH = "data/train50.csv"
VAL_PATH   = "data/val50.csv"
TEST_PATH  = "data/test50.csv"

RANDOM_SEED = 42

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

print(f"Total proteins: {len(df)}")

# -----------------------------
# Step 1: Train+Val vs Test
# -----------------------------
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=RANDOM_SEED,
    shuffle=True
)

# -----------------------------
# Step 2: Train vs Val
# -----------------------------
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.176,  # ≈ 15% of full dataset
    random_state=RANDOM_SEED,
    shuffle=True
)

# -----------------------------
# Save splits
# -----------------------------
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# -----------------------------
# Stats
# -----------------------------
print("✅ Split complete")
print(f"Train proteins: {len(train_df)}")
print(f"Val proteins:   {len(val_df)}")
print(f"Test proteins:  {len(test_df)}")
