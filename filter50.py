import pandas as pd
import ast
from collections import Counter
### phase 1.3: Filtering GO terms by frequency (≥ 50) ##
# Load propagated dataset
df = pd.read_csv("data/dataset_go_propagated.csv")

# Convert string lists back to lists
df["go_MF"] = df["go_MF"].apply(ast.literal_eval)
df["go_BP"] = df["go_BP"].apply(ast.literal_eval)
df["go_CC"] = df["go_CC"].apply(ast.literal_eval)

# Helper functions
def compute_go_frequency(go_lists):
    counter = Counter()
    for terms in go_lists:
        counter.update(terms)
    return counter

def filter_go_terms(go_lists, min_freq):
    freq = compute_go_frequency(go_lists)
    allowed = {go for go, c in freq.items() if c >= min_freq}
    filtered = [[go for go in terms if go in allowed] for terms in go_lists]
    return filtered, allowed

# Apply filtering per GO type
df["go_MF"], mf_terms = filter_go_terms(df["go_MF"], min_freq=50)
df["go_BP"], bp_terms = filter_go_terms(df["go_BP"], min_freq=50)
df["go_CC"], cc_terms = filter_go_terms(df["go_CC"], min_freq=50)

# Remove proteins with no labels in any GO type
df = df[
    (df["go_MF"].str.len() > 0) |
    (df["go_BP"].str.len() > 0) |
    (df["go_CC"].str.len() > 0)
]

# Save final label-engineered dataset
df.to_csv("data/dataset_go_filtered.csv", index=False)

print("✅ Phase 1.3 complete: GO terms filtered (freq ≥ 50)")
print("Remaining MF terms:", len(mf_terms))
print("Remaining BP terms:", len(bp_terms))
print("Remaining CC terms:", len(cc_terms))
print("Remaining proteins:", len(df))
