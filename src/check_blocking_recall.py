import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import blocking
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"

print("Connecting and building views...")
con = blocking.create_connection()
blocking.create_views(
    con,
    TRAIN_DIR / "train_source1.tsv",
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv"
)

print("Running blocking query (no ground-truth help)...")
query = blocking.generate_blocked_pairs(con)
con.execute("CREATE OR REPLACE TEMP TABLE blocked_only AS " + query)

print("Recreating validation split...")
training_sample = pd.read_csv(BASE_DIR / "output" / "training_sample.tsv", sep="\t", dtype=str)
groups = training_sample["source1_entity_id"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(splitter.split(training_sample, training_sample["label"], groups=groups))
val_entities = list(training_sample.iloc[val_idx]["source1_entity_id"].unique())

print(f"Validation entities: {len(val_entities):,}")

print("Loading ground truth...")
gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["source1_entity_id"].isin(val_entities)]

print("Loading blocked pairs for validation entities...")
val_entities_set = set(val_entities)
blocked_df = con.execute("SELECT source1_entity_id, candidate_entity_id FROM blocked_only").df()
blocked_df = blocked_df[blocked_df["source1_entity_id"].isin(val_entities_set)]
blocked_pairs = set(zip(blocked_df["source1_entity_id"], blocked_df["candidate_entity_id"]))

print("Checking recall...")
total_true_pairs = 0
found_pairs = 0
entities_with_missed_match = 0

for row in gt.itertuples():
    ids = [x.strip() for x in row.matched_entity_ids.split(",") if x.strip()] if row.matched_entity_ids else []
    entity_missed = False
    for cid in ids:
        total_true_pairs += 1
        if (row.source1_entity_id, cid) in blocked_pairs:
            found_pairs += 1
        else:
            entity_missed = True
    if entity_missed:
        entities_with_missed_match += 1

print()
print("=" * 50)
print("BLOCKING RECALL CHECK (real, no cheating)")
print("=" * 50)
print(f"Total true positive pairs: {total_true_pairs:,}")
print(f"Found by blocking alone:   {found_pairs:,}")
print(f"Blocking recall:           {found_pairs/total_true_pairs:.4f}")
print(f"Entities with at least 1 missed true match: {entities_with_missed_match:,}")
print("=" * 50)