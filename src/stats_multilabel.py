import pandas as pd
import ast

# Load the dataset
df = pd.read_csv('data/train_BP_propagated.csv')

# Number of proteins
num_proteins = len(df)

# Parse GO terms and get unique ones
all_go_terms = set()
total_go_count = 0
for go_list_str in df['go_BP']:
    # Convert string representation of list to actual list
    go_list = ast.literal_eval(go_list_str)
    all_go_terms.update(go_list)
    total_go_count += len(go_list)

num_unique_go_terms = len(all_go_terms)

# Print statistics
print("=" * 50)
print("Multilabel Dataset Statistics")
print("=" * 50)
print(f"Number of proteins: {num_proteins}")
print(f"Total GO terms: {total_go_count}")
print(f"Number of unique GO terms: {num_unique_go_terms}")
print("=" * 50)
