import csv
from collections import Counter

with open("data/labels.csv") as f:
    rows = list(csv.DictReader(f))

bad_ranges = [(488, 505), (514, 527)]

filtered = [
    row for row in rows
    if not any(start <= int(row["clip_id"]) <= end for start, end in bad_ranges)
]

directions = Counter(row["direction"] for row in filtered)
feet = Counter(row["foot"] for row in filtered)

print(f"Total clips after exclusion: {len(filtered)}")
print(f"Direction distribution: {directions}")
print(f"Foot distribution: {feet}")