import duckdb
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "output"

CANDIDATE_FILE = OUTPUT_DIR / "training_candidate_pairs.tsv"
OUTPUT_FILE = OUTPUT_DIR / "training_sample.tsv"

TEMP_DIR = OUTPUT_DIR / "duckdb_tmp"
TEMP_DIR.mkdir(exist_ok=True)


# =========================================================
# HELPER
# =========================================================

def sql_path(path):
    """
    Convert Windows path into a safe SQL string literal.
    """
    return str(path).replace("\\", "/").replace("'", "''")


# =========================================================
# CONNECTION
# =========================================================

con = duckdb.connect()

con.execute("SET memory_limit='4GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute(
    f"SET temp_directory='{sql_path(TEMP_DIR)}'"
)


# =========================================================
# INPUT FILES
# =========================================================

s1_path = TRAIN_DIR / "train_source1.tsv"
s2_path = TRAIN_DIR / "train_source2.tsv"
s3_path = TRAIN_DIR / "train_source3.tsv"
gt_path = TRAIN_DIR / "train_ground_truth.tsv"


print("=" * 60)
print("TRAINING SAMPLE GENERATION")
print("=" * 60)


# =========================================================
# STEP 1 — READ CANDIDATES + GROUND TRUTH
# =========================================================

print()
print("Reading candidate pairs and ground truth...")


con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE labeled_candidates AS

    WITH candidates AS (

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM read_csv(
            '{sql_path(CANDIDATE_FILE)}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
    ),

    ground_truth AS (

        SELECT
            source1_entity_id,
            matched_entity_ids

        FROM read_csv(
            '{sql_path(gt_path)}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
    )

    SELECT

        c.source1_entity_id,

        c.candidate_entity_id,

        CASE

            WHEN
                ',' ||
                coalesce(g.matched_entity_ids, '') ||
                ','
                LIKE
                '%,' ||
                c.candidate_entity_id ||
                ',%'

            THEN 1

            ELSE 0

        END AS label

    FROM candidates c

    LEFT JOIN ground_truth g

        ON c.source1_entity_id =
           g.source1_entity_id
    """
)


# =========================================================
# STEP 2 — LABEL COUNTS
# =========================================================

print()
print("Checking labels...")


counts = con.execute(
    """
    SELECT
        label,
        COUNT(*) AS count

    FROM labeled_candidates

    GROUP BY label

    ORDER BY label
    """
).fetchall()


for label, count in counts:

    print(
        f"Label {label}: {count:,}"
    )


# =========================================================
# STEP 3 — SAMPLE
# =========================================================

print()
print("Selecting training sample...")
print("Target: approximately 250,000 positives")
print("        approximately 250,000 negatives")


con.execute(
    """
    CREATE OR REPLACE TEMP TABLE training_sample_pairs AS

    SELECT
        source1_entity_id,
        candidate_entity_id,
        label

    FROM labeled_candidates

    WHERE

        (
            label = 1

            AND

            mod(
                hash(
                    source1_entity_id ||
                    '|' ||
                    candidate_entity_id
                ),
                30
            ) = 0
        )

        OR

        (
            label = 0

            AND

            mod(
                hash(
                    source1_entity_id ||
                    '|' ||
                    candidate_entity_id
                ),
                180
            ) = 0
        )
    """
)


# =========================================================
# STEP 4 — SAMPLE COUNTS
# =========================================================

sample_counts = con.execute(
    """
    SELECT
        label,
        COUNT(*) AS count

    FROM training_sample_pairs

    GROUP BY label

    ORDER BY label
    """
).fetchall()


print()
print("Selected sample:")

total = 0

for label, count in sample_counts:

    print(
        f"Label {label}: {count:,}"
    )

    total += count


print(
    f"Total training pairs: {total:,}"
)


# =========================================================
# STEP 5 — CREATE SOURCE VIEWS
# =========================================================

print()
print("Preparing source tables...")


con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE source1 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{sql_path(s1_path)}',
        sep='\\t',
        header=true,
        all_varchar=true
    )
    """
)


con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE source2 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{sql_path(s2_path)}',
        sep='\\t',
        header=true,
        all_varchar=true
    )
    """
)


con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE source3 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{sql_path(s3_path)}',
        sep='\\t',
        header=true,
        all_varchar=true
    )
    """
)


# =========================================================
# STEP 6 — COMBINE SOURCE 2 + SOURCE 3
# =========================================================

print("Combining Source 2 and Source 3...")


con.execute(
    """
    CREATE OR REPLACE TEMP TABLE all_candidates AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM source2

    UNION ALL

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM source3
    """
)


# =========================================================
# STEP 7 — JOIN TRAINING SAMPLE TO RECORDS
# =========================================================

print()
print("Adding business record information...")
print("This may take a few minutes...")


con.execute(
    """
    CREATE OR REPLACE TEMP TABLE final_training_sample AS

    SELECT

        p.source1_entity_id,

        p.candidate_entity_id,

        s1.business_name
            AS source1_business_name,

        s1.business_address
            AS source1_business_address,

        s1.country
            AS source1_country,

        c.business_name
            AS candidate_business_name,

        c.business_address
            AS candidate_business_address,

        c.country
            AS candidate_country,

        p.label

    FROM training_sample_pairs p

    INNER JOIN source1 s1

        ON p.source1_entity_id =
           s1.entity_id

    INNER JOIN all_candidates c

        ON p.candidate_entity_id =
           c.entity_id
    """
)


# =========================================================
# STEP 8 — CHECK JOIN
# =========================================================

joined_count = con.execute(
    """
    SELECT COUNT(*)
    FROM final_training_sample
    """
).fetchone()[0]


print()
print(
    f"Joined training records: {joined_count:,}"
)


if joined_count == 0:

    raise RuntimeError(
        "ERROR: No training records were joined."
    )


# =========================================================
# STEP 9 — WRITE OUTPUT
# =========================================================

print()
print("Writing training_sample.tsv...")


output_sql = sql_path(OUTPUT_FILE)


con.execute(
    f"""
    COPY final_training_sample

    TO '{output_sql}'

    WITH (
        FORMAT CSV,
        DELIMITER '\\t',
        HEADER TRUE
    )
    """
)


# =========================================================
# STEP 10 — VERIFY FILE
# =========================================================

print()
print("Verifying output file...")


if not OUTPUT_FILE.exists():

    raise RuntimeError(
        "ERROR: training_sample.tsv was not created."
    )


file_size = OUTPUT_FILE.stat().st_size


print(
    f"Output size: {file_size:,} bytes"
)


print()
print("=" * 60)
print("TRAINING SAMPLE COMPLETE")
print("=" * 60)

print()
print(
    f"Training pairs: {joined_count:,}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()


con.close()