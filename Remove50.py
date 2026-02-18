import pandas as pd
import ast
### phase 0: Removing sequences with length < 50 ##
# Load merged dataset
df = pd.read_csv("data/multilabel_dataset_clean.csv")

# Compute sequence length
df["seq_len"] = df["sequence"].str.len()

print("Total proteins before filtering:", len(df))
print("Proteins <50 aa:", (df["seq_len"] < 50).sum())

# Remove short sequences
df_clean = df[df["seq_len"] >= 50].copy()

print("Total proteins after filtering:", len(df_clean))

# Drop helper column
df_clean = df_clean.drop(columns=["seq_len"])

# Save cleaned dataset
df_clean.to_csv("dataset_clean50.csv", index=False)

print("✅ Phase 0 complete: Short sequences removed")
