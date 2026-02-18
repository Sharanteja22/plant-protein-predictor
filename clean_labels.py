import pandas as pd
import ast

IN_PATH = "data/multilabel_dataset.csv"
OUT_PATH = "data/multilabel_dataset_clean.csv"

# Load dataset
df = pd.read_csv(IN_PATH)

# Convert string list → Python list
df["go_id"] = df["go_id"].apply(lambda x: ast.literal_eval(x))

# Deduplicate GO terms per protein
df["go_id"] = df["go_id"].apply(lambda terms: sorted(set(terms)))

# Remove proteins with no GO terms (just in case)
df = df[df["go_id"].map(len) > 0]

# Save cleaned dataset
df.to_csv(OUT_PATH, index=False)

print(f"✅ Cleaned dataset: {len(df)} proteins")
print("Sample:")
print(df.head())