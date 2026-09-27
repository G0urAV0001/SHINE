import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "entity_matcher.pkl"

print("Loading model...")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURE_COLUMNS = saved["feature_columns"]

# ---- Step 1: recreate the exact same validation split as train.py ----
print("Recreating validation split...")
training_sample = pd.read_csv(BASE_DIR / "output" / "training_sample.tsv", sep="\t", dtype=str)
groups = training_sample["source1_entity_id"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(splitter.split(training_sample, training_sample["label"], groups=groups))
val_entities = set(training_sample.iloc[val_idx]["source1_entity_id"].unique())
print(f"Validation entities: {len(val_entities):,}")

# ---- Step 2: load ground truth for those entities ----
print("Loading ground truth...")
gt = pd.read_csv(BASE_DIR / "data" / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["source1_entity_id"].isin(val_entities)]
true_matches = {}
for row in gt.itertuples():
    ids = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
    true_matches[row.source1_entity_id] = ids

# ---- Step 3: load full candidate pairs for validation entities ----
print("Loading candidate pairs...")
cand = pd.read_csv(BASE_DIR / "output" / "training_candidate_pairs.tsv", sep="\t", dtype=str)
cand = cand[cand["source1_entity_id"].isin(val_entities)]
print(f"Candidate pairs for validation entities: {len(cand):,}")

# ---- Step 4: load business records to build features ----
print("Loading source records...")
s1 = pd.read_csv(BASE_DIR / "data" / "train" / "train_source1.tsv", sep="\t", dtype=str).set_index("entity_id")
s2 = pd.read_csv(BASE_DIR / "data" / "train" / "train_source2.tsv", sep="\t", dtype=str).set_index("entity_id")
s3 = pd.read_csv(BASE_DIR / "data" / "train" / "train_source3.tsv", sep="\t", dtype=str).set_index("entity_id")
all_candidates = pd.concat([s2, s3])

cand = cand.join(s1[["business_name", "business_address", "country"]].add_prefix("source1_"), on="source1_entity_id")
cand = cand.join(all_candidates[["business_name", "business_address", "country"]].add_prefix("candidate_"), on="candidate_entity_id")
cand = cand.dropna(subset=["source1_business_name", "candidate_business_name"])

# ---- Step 5: compute features (same as train.py) ----
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower().strip()

print("Computing features...")
name1 = cand["source1_business_name"].map(clean_text)
name2 = cand["candidate_business_name"].map(clean_text)
address1 = cand["source1_business_address"].map(clean_text)
address2 = cand["candidate_business_address"].map(clean_text)
country1 = cand["source1_country"].map(clean_text)
country2 = cand["candidate_country"].map(clean_text)

cand["name_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(name1, name2)]
cand["name_token_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(name1, name2)]
cand["name_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(name1, name2)]
cand["address_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(address1, address2)]
cand["address_token_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(address1, address2)]
cand["address_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(address1, address2)]
cand["country_match"] = (country1 == country2).astype(np.int8)
cand["name_length_difference_norm"] = (name1.str.len() - name2.str.len()).abs() / (1 + name1.str.len() + name2.str.len())
cand["address_length_difference_norm"] = (address1.str.len() - address2.str.len()).abs() / (1 + address1.str.len() + address2.str.len())
cand["name_token_difference"] = (name1.str.split().str.len() - name2.str.split().str.len()).abs()
cand["address_token_difference"] = (address1.str.split().str.len() - address2.str.split().str.len()).abs()

# ---- Step 6: predict probabilities ----
print("Predicting probabilities...")
X = cand[FEATURE_COLUMNS]
cand["probability"] = model.predict_proba(X)[:, 1]

# ---- Step 7: entity-level F0.5 sweep ----
def entity_f05(threshold):
    preds = cand[cand["probability"] >= threshold]
    pred_by_entity = preds.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    scores = []
    for entity in val_entities:
        true_set = true_matches.get(entity, set())
        pred_set = pred_by_entity.get(entity, set())

        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue

        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0

        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue

        beta2 = 0.25
        denom = (beta2 * precision) + recall
        f05 = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        scores.append(f05)

    return np.mean(scores)

print()
print("Sweeping thresholds (entity-level F0.5)...")
print(f"{'Threshold':<12}{'F0.5':<10}")
best_threshold, best_score = None, -1
for t in np.arange(0.30, 0.995, 0.01):
    score = entity_f05(round(t, 2))
    print(f"{t:<12.2f}{score:<10.4f}")
    if score > best_score:
        best_score = score
        best_threshold = round(t, 2)

print()
print("=" * 40)
print(f"BEST THRESHOLD: {best_threshold}")
print(f"BEST ENTITY-LEVEL F0.5: {best_score:.4f}")
print("=" * 40)