import pandas as pd
##Grouping GO terms per protein##
IN_PATH = "data/clean_dataset.csv"
OUT_PATH = "data/multilabel_dataset.csv"


# Step 1: Load merged dataset
df = pd.read_csv(IN_PATH)

print(f"Loaded {len(df)} rows, {df['protein_id'].nunique()} unique proteins")

# Step 2: Group GO terms per protein
grouped = df.groupby(["protein_id", "sequence"])["go_id"].apply(list).reset_index()

# Step 3: Save collapsed dataset
grouped.to_csv(OUT_PATH, index=False)

print(f"✅ Saved {len(grouped)} proteins to {OUT_PATH}")
print("Sample:")
print(grouped.head())