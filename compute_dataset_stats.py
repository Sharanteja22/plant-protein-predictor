#!/usr/bin/env python3
import csv
from collections import defaultdict

path = 'data/dataset_clean50.csv'
output_path = 'data/sequence_length_distribution.csv'

length_counts = defaultdict(int)

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        seq = row[1]
        length = len(seq)
        length_counts[length] += 1

# Compute ranges
bin_size = 50
range_counts = defaultdict(int)
max_len = max(length_counts.keys())
for length in length_counts:
    bin_start = ((length - 1) // bin_size) * bin_size + 1
    bin_end = bin_start + bin_size - 1
    range_label = f"{bin_start}-{bin_end}"
    range_counts[range_label] += length_counts[length]

total_proteins = sum(length_counts.values())

# Write to CSV
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['length', 'count'])
    for length in sorted(length_counts.keys()):
        writer.writerow([length, length_counts[length]])
    writer.writerow([])  # blank line
    writer.writerow(['range', 'count'])
    for range_label in sorted(range_counts.keys()):
        writer.writerow([range_label, range_counts[range_label]])
    writer.writerow(['Total Proteins', total_proteins])

print(f"Sequence length distribution and ranges saved to {output_path}")
print(f"Total Proteins: {total_proteins}")



