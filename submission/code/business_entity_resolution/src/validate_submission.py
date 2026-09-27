from pathlib import Path
import pandas as pd
import sys

# ============================================================
# AMAZON BUSINESS ENTITY RESOLUTION
# FINAL SUBMISSION VALIDATOR
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RESULT_FILE = BASE / "output" / "matching_results.tsv"
CANDIDATE_FILE = BASE / "output" / "candidate_pairs.tsv"

TEST_S1 = BASE / "data" / "test" / "test_source1.tsv"
TEST_S2 = BASE / "data" / "test" / "test_source2.tsv"
TEST_S3 = BASE / "data" / "test" / "test_source3.tsv"

print("=" * 65)
print("AMAZON BUSINESS ENTITY RESOLUTION")
print("FINAL SUBMISSION VALIDATION")
print("=" * 65)

errors = []
warnings = []


# ------------------------------------------------------------
# 1. CHECK FILES
# ------------------------------------------------------------

print("\n[1/7] Checking required files...")

required_files = [
    RESULT_FILE,
    CANDIDATE_FILE,
    TEST_S1,
    TEST_S2,
    TEST_S3,
]

for f in required_files:
    if f.exists():
        print(f"  OK: {f.relative_to(BASE)}")
    else:
        errors.append(f"Missing file: {f}")
        print(f"  ERROR: {f}")


if errors:
    print("\nVALIDATION FAILED.")
    for e in errors:
        print(" -", e)
    sys.exit(1)


# ------------------------------------------------------------
# 2. LOAD TEST ENTITY IDS
# ------------------------------------------------------------

print("\n[2/7] Loading test entity IDs...")

s1_ids = set(
    pd.read_csv(
        TEST_S1,
        sep="\t",
        dtype=str,
        usecols=["entity_id"]
    )["entity_id"]
)

s2_ids = set(
    pd.read_csv(
        TEST_S2,
        sep="\t",
        dtype=str,
        usecols=["entity_id"]
    )["entity_id"]
)

s3_ids = set(
    pd.read_csv(
        TEST_S3,
        sep="\t",
        dtype=str,
        usecols=["entity_id"]
    )["entity_id"]
)

valid_match_ids = s2_ids | s3_ids

print(f"  Test Source 1 IDs: {len(s1_ids):,}")
print(f"  Test Source 2 IDs: {len(s2_ids):,}")
print(f"  Test Source 3 IDs: {len(s3_ids):,}")
print(f"  Valid match IDs:    {len(valid_match_ids):,}")


# ------------------------------------------------------------
# 3. CHECK MATCHING RESULTS
# ------------------------------------------------------------

print("\n[3/7] Checking matching_results.tsv...")

results = pd.read_csv(
    RESULT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

expected_result_columns = [
    "source1_entity_id",
    "matched_entity_ids"
]

if list(results.columns) != expected_result_columns:
    errors.append(
        f"Wrong matching_results columns: {list(results.columns)}"
    )
    print("  ERROR: Wrong columns")
else:
    print("  OK: Columns are correct")


# Row count
if len(results) != len(s1_ids):
    errors.append(
        f"Expected {len(s1_ids):,} result rows, got {len(results):,}"
    )
    print("  ERROR: Wrong row count")
else:
    print(f"  OK: Row count = {len(results):,}")


# Duplicate S1 IDs
duplicate_s1 = results["source1_entity_id"].duplicated().sum()

if duplicate_s1 != 0:
    errors.append(f"Duplicate S1 IDs: {duplicate_s1}")
    print(f"  ERROR: Duplicate S1 IDs = {duplicate_s1}")
else:
    print("  OK: No duplicate S1 IDs")


# Missing / extra S1 IDs
result_s1_ids = set(results["source1_entity_id"])

missing_s1 = s1_ids - result_s1_ids
extra_s1 = result_s1_ids - s1_ids

if missing_s1:
    errors.append(f"Missing S1 IDs: {len(missing_s1):,}")
    print(f"  ERROR: Missing S1 IDs = {len(missing_s1):,}")
else:
    print("  OK: No missing S1 IDs")

if extra_s1:
    errors.append(f"Invalid extra S1 IDs: {len(extra_s1):,}")
    print(f"  ERROR: Extra S1 IDs = {len(extra_s1):,}")
else:
    print("  OK: No extra S1 IDs")


# ------------------------------------------------------------
# 4. CHECK MATCH VALUES
# ------------------------------------------------------------

print("\n[4/7] Checking predicted match IDs...")

invalid_match_count = 0
duplicate_match_count = 0
total_matches = 0
entities_with_matches = 0
entities_without_matches = 0

for row in results.itertuples(index=False):

    match_text = row.matched_entity_ids

    if not match_text:
        entities_without_matches += 1
        continue

    entities_with_matches += 1

    matches = match_text.split(",")

    # Count matches
    total_matches += len(matches)

    # Duplicate IDs inside same row
    if len(matches) != len(set(matches)):
        duplicate_match_count += 1

    # Invalid IDs
    for match_id in matches:
        if match_id not in valid_match_ids:
            invalid_match_count += 1
            if invalid_match_count <= 5:
                print(
                    f"  Invalid match example: "
                    f"{row.source1_entity_id} -> {match_id}"
                )


if invalid_match_count != 0:
    errors.append(
        f"Invalid S2/S3 match IDs: {invalid_match_count}"
    )
    print(
        f"  ERROR: Invalid match IDs = "
        f"{invalid_match_count:,}"
    )
else:
    print("  OK: Every predicted ID belongs to test Source 2/3")


if duplicate_match_count != 0:
    errors.append(
        f"Rows containing duplicate match IDs: "
        f"{duplicate_match_count}"
    )
    print(
        f"  ERROR: Duplicate match IDs = "
        f"{duplicate_match_count:,}"
    )
else:
    print("  OK: No duplicate match IDs inside rows")


print(f"  Entities with matches:    {entities_with_matches:,}")
print(f"  Entities without matches: {entities_without_matches:,}")
print(f"  Total predicted matches:  {total_matches:,}")


# ------------------------------------------------------------
# 5. CHECK CANDIDATE FILE
# ------------------------------------------------------------

print("\n[5/7] Checking candidate_pairs.tsv...")

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

expected_candidate_columns = [
    "source1_entity_id",
    "candidate_entity_ids"
]

if list(candidates.columns) != expected_candidate_columns:
    errors.append(
        f"Wrong candidate columns: {list(candidates.columns)}"
    )
    print("  ERROR: Wrong columns")
else:
    print("  OK: Columns are correct")


if len(candidates) != len(s1_ids):
    errors.append(
        f"Expected {len(s1_ids):,} candidate rows, "
        f"got {len(candidates):,}"
    )
    print("  ERROR: Wrong candidate row count")
else:
    print(f"  OK: Candidate rows = {len(candidates):,}")


candidate_duplicate_s1 = (
    candidates["source1_entity_id"].duplicated().sum()
)

if candidate_duplicate_s1 != 0:
    errors.append(
        f"Duplicate S1 IDs in candidates: "
        f"{candidate_duplicate_s1}"
    )
    print(
        f"  ERROR: Duplicate candidate S1 IDs = "
        f"{candidate_duplicate_s1}"
    )
else:
    print("  OK: No duplicate candidate S1 IDs")


# ------------------------------------------------------------
# 6. CHECK EVERY PREDICTED MATCH IS A CANDIDATE
# ------------------------------------------------------------

print("\n[6/7] Checking candidate consistency...")
print("  This may take a little while...")

candidate_lookup = dict(
    zip(
        candidates["source1_entity_id"],
        candidates["candidate_entity_ids"]
    )
)

outside_candidate_count = 0

for row in results.itertuples(index=False):

    if not row.matched_entity_ids:
        continue

    candidate_text = candidate_lookup.get(
        row.source1_entity_id,
        ""
    )

    candidate_set = set(
        candidate_text.split(",")
    ) if candidate_text else set()

    predicted_set = set(
        row.matched_entity_ids.split(",")
    )

    outside = predicted_set - candidate_set

    if outside:
        outside_candidate_count += 1

        if outside_candidate_count <= 5:
            print(
                f"  Example outside candidate set: "
                f"{row.source1_entity_id} -> "
                f"{list(outside)[:3]}"
            )


if outside_candidate_count != 0:
    errors.append(
        f"Rows with predictions outside candidate set: "
        f"{outside_candidate_count}"
    )
    print(
        f"  ERROR: Predictions outside candidate set = "
        f"{outside_candidate_count:,}"
    )
else:
    print(
        "  OK: Every predicted match is inside "
        "candidate_pairs.tsv"
    )


# ------------------------------------------------------------
# 7. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

print(f"Source 1 test entities:       {len(s1_ids):,}")
print(f"Result rows:                  {len(results):,}")
print(f"Candidate rows:               {len(candidates):,}")
print(f"Entities with matches:        {entities_with_matches:,}")
print(f"Entities without matches:     {entities_without_matches:,}")
print(f"Total predicted matches:      {total_matches:,}")
print(f"Invalid match IDs:            {invalid_match_count:,}")
print(f"Duplicate match rows:         {duplicate_match_count:,}")
print(f"Outside candidate set:        {outside_candidate_count:,}")

print("=" * 65)

if errors:
    print("\nVALIDATION FAILED")
    print("\nProblems found:")
    for e in errors:
        print(" -", e)

    sys.exit(1)

else:
    print("\nALL VALIDATION CHECKS PASSED")
    print("\nYour output files are structurally valid and")
    print("ready for the submission-packaging stage.")
    print("=" * 65)