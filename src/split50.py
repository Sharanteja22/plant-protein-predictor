import pandas as pd
import ast
from goatools.obo_parser import GODag
### phase 1.1: Splitting GO terms into MF, BP, CC ##
# Load GO ontology
go_dag = GODag("go-basic.obo")

# Load cleaned dataset
df = pd.read_csv("data/dataset_clean50.csv")
df["go_id"] = df["go_id"].apply(ast.literal_eval)

# Helper: split GO terms by namespace
def split_go_terms(go_terms):
    mf, bp, cc = [], [], []
    for go in go_terms:
        if go in go_dag:
            ns = go_dag[go].namespace
            if ns == "molecular_function":
                mf.append(go)
            elif ns == "biological_process":
                bp.append(go)
            elif ns == "cellular_component":
                cc.append(go)
    return mf, bp, cc

# Apply split
df["go_MF"], df["go_BP"], df["go_CC"] = zip(
    *df["go_id"].apply(split_go_terms)
)


# Save intermediate file
df.to_csv("data/dataset_split_go_types.csv", index=False)

print("✅ Phase 1.1 complete: GO terms split into MF / BP / CC")
print("MF proteins:", (df["go_MF"].str.len() > 0).sum())
print("BP proteins:", (df["go_BP"].str.len() > 0).sum())
print("CC proteins:", (df["go_CC"].str.len() > 0).sum())
