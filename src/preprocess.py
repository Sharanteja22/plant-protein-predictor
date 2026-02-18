from Bio import SeqIO
import pandas as pd
#### Phase 1.1: Building clean dataset from Araport11 FASTA and ATH GO SLIM annotation ##
FASTA_PATH = "data/Araport11_genes.201606.pep.fasta"
GO_PATH = "data/ATH_GO_GOSLIM.txt"
OUT_PATH = "data/clean_dataset.csv"

# Parse FASTA
seq_records = []
for rec in SeqIO.parse(FASTA_PATH, "fasta"):
    gene_id = rec.id.split('.')[0]   # AT1G01010.1 → AT1G01010
    seq_records.append({"protein_id": gene_id, "sequence": str(rec.seq)})
seq_df = pd.DataFrame(seq_records).drop_duplicates("protein_id")

# Parse GO file
go_df = pd.read_csv(GO_PATH, sep="\t", comment="!", header=None)

# Gene ID = col[0], GO ID = col[5]
go_df = go_df[[0, 5]]
go_df.columns = ["protein_id", "go_id"]

# Merge
merged = pd.merge(seq_df, go_df, on="protein_id", how="inner")

# Save
merged.to_csv(OUT_PATH, index=False)
print(f"✅ Saved {len(merged)} rows to {OUT_PATH}")