import csv
import joblib
import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
RESULT_FILE = OUTPUT_DIR / "matching_results.tsv"

MODEL_FILE = MODEL_DIR / "entity_matcher.pkl"
THRESHOLD_FILE = MODEL_DIR / "best_threshold.txt"

S1_FILE = BASE_DIR / "data" / "test" / "test_source1.tsv"
S2_FILE = BASE_DIR / "data" / "test" / "test_source2.tsv"
S3_FILE = BASE_DIR / "data" / "test" / "test_source3.tsv"


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 50_000


# =========================================================
# TEXT
# =========================================================

import unicodedata

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Strip accents (é -> e, ç -> c, etc.)
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(c for c in value if not unicodedata.combining(c))

    return value
# =========================================================
# FEATURES
# =========================================================

def calculate_features(df):

    name1 = df["source1_business_name"].map(clean_text)
    name2 = df["candidate_business_name"].map(clean_text)

    address1 = df["source1_business_address"].map(clean_text)
    address2 = df["candidate_business_address"].map(clean_text)

    country1 = df["source1_country"].map(clean_text)
    country2 = df["candidate_country"].map(clean_text)

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

    df["name_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    df["name_token_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    df["name_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(name1, name2)
    ]

    # -----------------------------------------------------
    # ADDRESS
    # -----------------------------------------------------

    df["address_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    df["address_token_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    df["address_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(address1, address2)
    ]

    # -----------------------------------------------------
    # COUNTRY
    # -----------------------------------------------------

    df["country_match"] = (
        country1 == country2
    ).astype(np.int8)

    # -----------------------------------------------------
    # STRUCTURAL FEATURES
    # -----------------------------------------------------

    name_len1 = name1.str.len()
    name_len2 = name2.str.len()

    address_len1 = address1.str.len()
    address_len2 = address2.str.len()

    df["name_length_difference_norm"] = (
        (name_len1 - name_len2).abs()
        /
        (1 + name_len1 + name_len2)
    )

    df["address_length_difference_norm"] = (
        (address_len1 - address_len2).abs()
        /
        (1 + address_len1 + address_len2)
    )

    df["name_token_difference"] = (
        name1.str.split().str.len()
        -
        name2.str.split().str.len()
    ).abs()

    df["address_token_difference"] = (
        address1.str.split().str.len()
        -
        address2.str.split().str.len()
    ).abs()

    return df


# =========================================================
# MODEL FEATURES
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
# LOAD MODEL
# =========================================================

print("=" * 60)
print("AMAZON BUSINESS ENTITY RESOLUTION")
print("FINAL PREDICTION")
print("=" * 60)

print()
print("Loading model...")

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]

print(
    f"Model loaded: {MODEL_FILE}"
)


# =========================================================
# LOAD THRESHOLD
# =========================================================

with open(
    THRESHOLD_FILE,
    "r"
) as f:

    threshold = float(
        f.read().strip()
    )

print(
    f"Decision threshold: {threshold:.2f}"
)


# =========================================================
# LOAD TEST SOURCES
# =========================================================

print()
print("Loading test source records...")


source1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

source2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str
)

source3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str
)


print(
    f"Source 1 records: {len(source1):,}"
)

print(
    f"Source 2 records: {len(source2):,}"
)

print(
    f"Source 3 records: {len(source3):,}"
)


# =========================================================
# BUILD LOOKUPS
# =========================================================

print()
print("Building record lookups...")


source1_lookup = source1.set_index(
    "entity_id"
).to_dict("index")


source2_lookup = source2.set_index(
    "entity_id"
).to_dict("index")


source3_lookup = source3.set_index(
    "entity_id"
).to_dict("index")


# One combined lookup for S2 + S3.

candidate_lookup = {}

candidate_lookup.update(
    source2_lookup
)

candidate_lookup.update(
    source3_lookup
)


# =========================================================
# PREPARE OUTPUT
# =========================================================

# Remove previous result if it exists.

if RESULT_FILE.exists():

    RESULT_FILE.unlink()


# We first collect predicted matches per S1.

matches = {}

for entity_id in source1["entity_id"]:

    matches[entity_id] = []


# =========================================================
# PROCESS CANDIDATES IN CHUNKS
# =========================================================

print()
print(
    f"Processing candidate file in chunks of "
    f"{CHUNK_SIZE:,}..."
)

print()
print("This can take some time because there are")
print("35.9 million candidate pairs.")
print()


processed = 0
predicted_matches = 0


candidate_reader = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)


for chunk_number, candidate_chunk in enumerate(
    candidate_reader,
    start=1
):

    # -----------------------------------------------------
    # Expand comma-separated candidates
    # -----------------------------------------------------

    rows = []

    for row in candidate_chunk.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id
        candidate_ids = row.candidate_entity_ids

        if (
            pd.isna(candidate_ids)
            or not candidate_ids
        ):
            continue

        s1_record = source1_lookup.get(
            s1_id
        )

        if s1_record is None:
            continue

        for candidate_id in candidate_ids.split(","):

            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            candidate_record = candidate_lookup.get(
                candidate_id
            )

            if candidate_record is None:
                continue

            rows.append({

                "source1_entity_id": s1_id,

                "candidate_entity_id": candidate_id,

                "source1_business_name":
                    s1_record["business_name"],

                "source1_business_address":
                    s1_record["business_address"],

                "source1_country":
                    s1_record["country"],

                "candidate_business_name":
                    candidate_record["business_name"],

                "candidate_business_address":
                    candidate_record["business_address"],

                "candidate_country":
                    candidate_record["country"],
            })


    # -----------------------------------------------------
    # Empty chunk
    # -----------------------------------------------------

    if not rows:

        processed += len(candidate_chunk)

        continue


    feature_df = pd.DataFrame(rows)


    # -----------------------------------------------------
    # Calculate same features used during training
    # -----------------------------------------------------

    feature_df = calculate_features(
        feature_df
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        feature_df[FEATURE_COLUMNS]
    )[:, 1]


    # -----------------------------------------------------
    # Apply learned threshold
    # -----------------------------------------------------

    positive_mask = (
        probabilities >= threshold
    )


    positive_rows = feature_df.loc[
        positive_mask,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ]


    # -----------------------------------------------------
    # Store predictions
    # -----------------------------------------------------

    for prediction in positive_rows.itertuples(
        index=False
    ):

        s1_id = prediction.source1_entity_id
        candidate_id = prediction.candidate_entity_id

        matches[s1_id].append(
            candidate_id
        )

        predicted_matches += 1


    processed += len(candidate_chunk)


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if chunk_number % 10 == 0:

        print(
            f"Chunks processed: {chunk_number:,} | "
            f"Candidate rows: {processed:,} | "
            f"Predicted matches: {predicted_matches:,}"
        )


# =========================================================
# REMOVE DUPLICATES
# =========================================================

print()
print("Removing duplicate predictions...")


for s1_id in matches:

    matches[s1_id] = sorted(
        set(matches[s1_id])
    )


# =========================================================
# WRITE FINAL RESULTS
# =========================================================

print()
print("Writing matching_results.tsv...")


with open(
    RESULT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as output:

    writer = csv.writer(
        output,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    for s1_id in source1["entity_id"]:

        matched_ids = matches[s1_id]

        writer.writerow([
            s1_id,
            ",".join(matched_ids)
        ])


# =========================================================
# FINAL STATISTICS
# =========================================================

nonempty = sum(
    1
    for ids in matches.values()
    if ids
)

singleton = len(matches) - nonempty

total_matches = sum(
    len(ids)
    for ids in matches.values()
)


print()
print("=" * 60)
print("FINAL PREDICTION COMPLETE")
print("=" * 60)

print(
    f"Source 1 entities: {len(matches):,}"
)

print(
    f"Entities with matches: {nonempty:,}"
)

print(
    f"Entities with no matches: {singleton:,}"
)

print(
    f"Total predicted matches: {total_matches:,}"
)

print()
print(
    f"Saved to: {RESULT_FILE}"
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)