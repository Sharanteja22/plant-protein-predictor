import pandas as pd
import ast
from goatools.obo_parser import GODag
### phase 1.2: Applying True Path Rule to propagate GO terms ##
# Load ontology
go_dag = GODag("go-basic.obo")

# Load dataset with split GO types
df = pd.read_csv("data/dataset_split_go_types.csv")

# Convert string lists back to Python lists
df["go_MF"] = df["go_MF"].apply(ast.literal_eval)
df["go_BP"] = df["go_BP"].apply(ast.literal_eval)
df["go_CC"] = df["go_CC"].apply(ast.literal_eval)

# True Path Rule propagation
def propagate_go_terms(go_terms):
    propagated = set(go_terms)
    for go in go_terms:
        if go in go_dag:
            parents = go_dag[go].get_all_parents()
            propagated.update(parents)
    return list(propagated)

# Apply propagation per GO type
df["go_MF"] = df["go_MF"].apply(propagate_go_terms)
df["go_BP"] = df["go_BP"].apply(propagate_go_terms)
df["go_CC"] = df["go_CC"].apply(propagate_go_terms)

# Save propagated dataset
df.to_csv("data/dataset_go_propagated.csv", index=False)

print("✅ Phase 1.2 complete: True Path Rule applied")



###split_go_types--->dataset_go_propagated.csv--->dataset_go_filterd.csv