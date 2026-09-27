import pandas as pd
import numpy as np
import joblib
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "training_sample.tsv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "entity_matcher.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.txt"


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Strip accents (é -> e, ç -> c, etc.)
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(c for c in value if not unicodedata.combining(c))

    return value
# =========================================================
# FEATURE CALCULATION
# =========================================================

def calculate_features(df):

    print("Calculating name similarities...")

    name1 = df["source1_business_name"].map(clean_text)
    name2 = df["candidate_business_name"].map(clean_text)

    print("  - name ratio")

    df["name_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    print("  - name token ratio")

    df["name_token_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    print("  - name partial ratio")

    df["name_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    print("Calculating address similarities...")

    address1 = df["source1_business_address"].map(clean_text)
    address2 = df["candidate_business_address"].map(clean_text)

    print("  - address ratio")

    df["address_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    print("  - address token ratio")

    df["address_token_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    print("  - address partial ratio")

    df["address_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    print("Calculating country features...")

    country1 = df["source1_country"].map(clean_text)
    country2 = df["candidate_country"].map(clean_text)

    df["country_match"] = (
        country1 == country2
    ).astype(np.int8)

    print("Calculating structural features...")

    df["name_length_difference"] = (
        name1.str.len() -
        name2.str.len()
    ).abs()

    df["address_length_difference"] = (
        address1.str.len() -
        address2.str.len()
    ).abs()

    df["name_token_difference"] = (
        name1.str.split().str.len() -
        name2.str.split().str.len()
    ).abs()

    df["address_token_difference"] = (
        address1.str.split().str.len() -
        address2.str.split().str.len()
    ).abs()

    # Normalize length differences so the model
    # does not get dominated by raw character counts.
    df["name_length_difference_norm"] = (
        df["name_length_difference"] /
        (
            1 +
            name1.str.len() +
            name2.str.len()
        )
    )

    df["address_length_difference_norm"] = (
        df["address_length_difference"] /
        (
            1 +
            address1.str.len() +
            address2.str.len()
        )
    )

    return df


# =========================================================
# FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [

    "name_ratio",
    "name_token_ratio",
    "name_partial_ratio",

    "address_ratio",
    "address_token_ratio",
    "address_partial_ratio",

    "country_match",

    "name_length_difference_norm",
    "address_length_difference_norm",

    "name_token_difference",
    "address_token_difference",
]


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("ENTITY MATCHING MODEL TRAINING")
print("=" * 60)

print()
print("Loading training sample...")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str
)

print(
    f"Training rows loaded: {len(df):,}"
)


# =========================================================
# FEATURES
# =========================================================

print()
print("Generating ML features...")

df = calculate_features(df)

print()
print("Feature generation complete.")


# =========================================================
# TARGET
# =========================================================

df["label"] = df["label"].astype(int)


# =========================================================
# GROUPED TRAIN / VALIDATION SPLIT
# =========================================================
#
# Important:
# We split by Source 1 entity rather than randomly
# splitting individual pairs.
#
# This prevents the same Source 1 entity appearing
# in both training and validation.
# =========================================================

print()
print("Creating entity-level validation split...")

groups = df["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        df,
        df["label"],
        groups=groups
    )
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

print(
    f"Training pairs:   {len(train_df):,}"
)

print(
    f"Validation pairs: {len(val_df):,}"
)

print(
    f"Training entities: "
    f"{train_df['source1_entity_id'].nunique():,}"
)

print(
    f"Validation entities: "
    f"{val_df['source1_entity_id'].nunique():,}"
)


# =========================================================
# MODEL
# =========================================================

print()
print("Training Random Forest...")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df["label"]


model = RandomForestClassifier(

    n_estimators=200,

    max_depth=14,

    min_samples_leaf=2,

    min_samples_split=5,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Model training complete.")


# =========================================================
# VALIDATION PROBABILITIES
# =========================================================

print()
print("Calculating validation probabilities...")

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


# =========================================================
# THRESHOLD SEARCH
# =========================================================
#
# F0.5 is precision-heavy in the challenge.
#
# We search several thresholds and choose the one
# producing the best validation F0.5.
# =========================================================

print()
print("Searching decision threshold...")

best_threshold = 0.50
best_f05 = -1

results = []

for threshold in np.arange(
    0.30,
    0.96,
    0.02
):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f05
        )
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# =========================================================
# DISPLAY BEST VALIDATION RESULT
# =========================================================

best_result = max(
    results,
    key=lambda x: x[3]
)

print()
print("=" * 60)
print("BEST VALIDATION RESULT")
print("=" * 60)

print(
    f"Threshold : {best_result[0]:.2f}"
)

print(
    f"Precision : {best_result[1]:.4f}"
)

print(
    f"Recall    : {best_result[2]:.4f}"
)

print(
    f"F0.5      : {best_result[3]:.4f}"
)


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("Saving model...")

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURE_COLUMNS
    },
    MODEL_PATH
)


# =========================================================
# SAVE THRESHOLD
# =========================================================

with open(
    THRESHOLD_PATH,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    f"Threshold saved to: {THRESHOLD_PATH}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print()
print("Feature importance:")

importance = sorted(
    zip(
        FEATURE_COLUMNS,
        model.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True
)

for feature, value in importance:

    print(
        f"{feature:40s} "
        f"{value:.4f}"
    )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)