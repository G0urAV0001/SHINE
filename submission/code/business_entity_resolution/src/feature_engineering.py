import pandas as pd
import re
from difflib import SequenceMatcher


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# SIMILARITY FUNCTIONS
# ---------------------------------------------------------

def sequence_similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_similarity(a, b):
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union else 0.0


def exact_match(a, b):
    if not a or not b:
        return 0

    return int(a == b)


# ---------------------------------------------------------
# NAME FEATURES
# ---------------------------------------------------------

def name_features(name1, name2):

    name1 = normalize_text(name1)
    name2 = normalize_text(name2)

    seq = sequence_similarity(name1, name2)
    token = token_similarity(name1, name2)
    exact = exact_match(name1, name2)

    return {
        "name_sequence_similarity": seq,
        "name_token_similarity": token,
        "name_exact_match": exact,
        "name_combined_similarity": (
            0.6 * seq + 0.4 * token
        )
    }


# ---------------------------------------------------------
# ADDRESS FEATURES
# ---------------------------------------------------------

def address_features(address1, address2):

    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    seq = sequence_similarity(address1, address2)
    token = token_similarity(address1, address2)
    exact = exact_match(address1, address2)

    return {
        "address_sequence_similarity": seq,
        "address_token_similarity": token,
        "address_exact_match": exact,
        "address_combined_similarity": (
            0.4 * seq + 0.6 * token
        )
    }


# ---------------------------------------------------------
# COUNTRY FEATURES
# ---------------------------------------------------------

def country_features(country1, country2):

    country1 = normalize_text(country1)
    country2 = normalize_text(country2)

    return {
        "country_exact_match": exact_match(
            country1,
            country2
        )
    }


# ---------------------------------------------------------
# ADDITIONAL STRUCTURAL FEATURES
# ---------------------------------------------------------

def structural_features(name1, name2, address1, address2):

    name1 = normalize_text(name1)
    name2 = normalize_text(name2)

    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    return {
        "name_length_difference": abs(
            len(name1) - len(name2)
        ),

        "address_length_difference": abs(
            len(address1) - len(address2)
        ),

        "name_token_count_difference": abs(
            len(name1.split()) -
            len(name2.split())
        ),

        "address_token_count_difference": abs(
            len(address1.split()) -
            len(address2.split())
        )
    }


# ---------------------------------------------------------
# CREATE FEATURES FOR ONE PAIR
# ---------------------------------------------------------

def create_pair_features(
    source1_row,
    source2_row
):

    name_feat = name_features(
        source1_row["business_name"],
        source2_row["business_name"]
    )

    address_feat = address_features(
        source1_row["business_address"],
        source2_row["business_address"]
    )

    country_feat = country_features(
        source1_row["country"],
        source2_row["country"]
    )

    structural_feat = structural_features(
        source1_row["business_name"],
        source2_row["business_name"],
        source1_row["business_address"],
        source2_row["business_address"]
    )

    features = {}

    features.update(name_feat)
    features.update(address_feat)
    features.update(country_feat)
    features.update(structural_feat)

    return features


# ---------------------------------------------------------
# CREATE FEATURES FOR MANY CANDIDATE PAIRS
# ---------------------------------------------------------

def create_candidate_features(
    source1_df,
    source2_df,
    candidate_df
):

    source1_lookup = source1_df.set_index(
        "entity_id"
    ).to_dict("index")

    source2_lookup = source2_df.set_index(
        "entity_id"
    ).to_dict("index")

    feature_rows = []

    for row in candidate_df.itertuples(index=False):

        source1_id = row.source1_entity_id
        candidate_ids = row.candidate_entity_ids

        if not candidate_ids:
            continue

        if source1_id not in source1_lookup:
            continue

        source1_row = source1_lookup[source1_id]

        for candidate_id in candidate_ids.split(","):

            if candidate_id not in source2_lookup:
                continue

            source2_row = source2_lookup[candidate_id]

            features = create_pair_features(
                source1_row,
                source2_row
            )

            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = candidate_id

            feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------
# GROUND TRUTH LABELING
# ---------------------------------------------------------

def add_ground_truth_labels(
    feature_df,
    ground_truth_df
):

    truth_lookup = {}

    for row in ground_truth_df.itertuples(index=False):

        source1_id = row.source1_entity_id
        matched_ids = str(row.matched_entity_ids)

        if matched_ids == "nan":
            matched_ids = ""

        matched_set = set()

        if matched_ids.strip():
            matched_set = set(
                x.strip()
                for x in matched_ids.split(",")
                if x.strip()
            )

        truth_lookup[source1_id] = matched_set

    labels = []

    for row in feature_df.itertuples(index=False):

        matched_set = truth_lookup.get(
            row.source1_entity_id,
            set()
        )

        labels.append(
            int(
                row.candidate_entity_id
                in matched_set
            )
        )

    result = feature_df.copy()
    result["label"] = labels

    return result


# ---------------------------------------------------------
# SMALL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Testing feature engineering...")
    print()

    source1 = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": [
            "ABC Technologies Private Limited"
        ],
        "business_address": [
            "123 MG Road Bangalore"
        ],
        "country": ["India"]
    })

    source2 = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002"],
        "business_name": [
            "ABC Technologies Pvt Ltd",
            "Random Company"
        ],
        "business_address": [
            "123 MG Rd Bangalore",
            "999 Road Mumbai"
        ],
        "country": [
            "India",
            "India"
        ]
    })

    candidates = pd.DataFrame({
        "source1_entity_id": ["S1-001"],
        "candidate_entity_ids": ["S2-001,S2-002"]
    })

    features = create_candidate_features(
        source1,
        source2,
        candidates
    )

    print("Generated features:")
    print(features.to_string(index=False))

    print()
    print("Feature columns:")
    print(list(features.columns))

    print()
    print("Feature engineering test completed successfully.")