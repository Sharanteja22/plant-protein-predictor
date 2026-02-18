import pandas as pd
# import ast

# # Read the CSV file
# df = pd.read_csv('data/dataset_go_filtered.csv')

# # Define the GO types
# types = ['MF', 'BP', 'CC']

# # Process each type
# for t in types:
#     col = f'go_{t}'
#     unique_go = set()
#     total_count = 0
    
#     for item in df[col]:
#         if pd.isna(item):
#             continue
#         # Parse the string list to actual list
#         go_list = ast.literal_eval(item)
#         unique_go.update(go_list)
#         total_count += len(go_list)
    
#     print(f"For {t}:")
#     print(f"  Number of unique GO terms: {len(unique_go)}")
#     print(f"  Total number of GO terms: {total_count}")
#     print()

train_ids = set(pd.read_csv("data/train50.csv")["protein_id"])
val_ids   = set(pd.read_csv("data/val50.csv")["protein_id"])

print("Overlap:", len(train_ids & val_ids))
