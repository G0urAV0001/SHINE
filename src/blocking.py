import duckdb
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

TEMP_DIR = OUTPUT_DIR / "duckdb_tmp"
TEMP_DIR.mkdir(exist_ok=True)


# =========================================================
# DUCKDB CONNECTION
# =========================================================

def create_connection():

    con = duckdb.connect()

    # Keep memory usage controlled.
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    # Allow DuckDB to spill temporary operations to disk.
    con.execute(
        f"SET temp_directory='{TEMP_DIR.as_posix()}'"
    )

    return con


# =========================================================
# SQL NORMALIZATION
# =========================================================

NAME_CLEAN = """
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                lower(coalesce(business_name, '')),
                                '[^a-z0-9]+', ' ', 'g'
                            ),
                            '\\bprivate limited\\b',
                            'pvt ltd',
                            'g'
                        ),
                        '\\bprivate ltd\\b',
                        'pvt ltd',
                        'g'
                    ),
                    '\\blimited\\b',
                    'ltd',
                    'g'
                ),
                '\\bcorporation\\b',
                'corp',
                'g'
            ),
            '\\bincorporated\\b',
            'inc',
            'g'
        )
    )
"""

ADDRESS_CLEAN = """
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                regexp_replace(
                                    regexp_replace(
                                        lower(coalesce(business_address, '')),
                                        '[^a-z0-9]+', ' ', 'g'
                                    ),
                                    '\\broad\\b',
                                    'rd',
                                    'g'
                                ),
                                '\\bstreet\\b',
                                'st',
                                'g'
                            ),
                            '\\bavenue\\b',
                            'ave',
                            'g'
                        ),
                        '\\bboulevard\\b',
                        'blvd',
                        'g'
                    ),
                    '\\bhighway\\b',
                    'hwy',
                    'g'
                ),
                '\\blane\\b',
                'ln',
                'g'
            ),
            '\\bdrive\\b',
            'dr',
            'g'
        )
    )
"""


# =========================================================
# CREATE NORMALIZED VIEWS
# =========================================================

def create_views(con, source1_path, source2_path, source3_path):

    print("Reading Source 1...")
    print("Reading Source 2...")
    print("Reading Source 3...")

    con.execute(f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,

            {NAME_CLEAN} AS name_clean,

            {ADDRESS_CLEAN} AS address_clean,

            lower(trim(coalesce(country, ''))) AS country_clean

        FROM read_csv(
            '{source1_path.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW s2 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,

            {NAME_CLEAN} AS name_clean,

            {ADDRESS_CLEAN} AS address_clean,

            lower(trim(coalesce(country, ''))) AS country_clean

        FROM read_csv(
            '{source2_path.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW s3 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,

            {NAME_CLEAN} AS name_clean,

            {ADDRESS_CLEAN} AS address_clean,

            lower(trim(coalesce(country, ''))) AS country_clean

        FROM read_csv(
            '{source3_path.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
    """)


# =========================================================
# BLOCKING QUERY
# =========================================================

def generate_blocked_pairs(con):

    print()
    print("Generating candidate pairs...")
    print("Blocking rules:")
    print("  1. Exact normalized name")
    print("  2. Exact normalized address")
    print("  3. Country + name prefix")
    print("  4. Country + address number")
    print("  5. Country + sorted name tokens")
    print("  6. Country + address without house number")
    print()

    query = f"""

    WITH

    -------------------------------------------------------
    -- SOURCE 1 BLOCK KEYS
    -------------------------------------------------------

    s1_keys AS (
        SELECT
            entity_id AS source1_entity_id,
            country_clean,

            name_clean,

            address_clean,

            country_clean || '|' ||
            left(
                regexp_replace(name_clean, '\\s+', '', 'g'),
                6
            ) AS name_prefix,

            country_clean || '|' ||
            coalesce(
                regexp_extract(address_clean, '(\\d+)', 1),
                ''
            ) AS address_number,

            country_clean || '|' ||
            array_to_string(
                list_sort(
                    string_split(name_clean, ' ')
                ),
                ''
            ) AS name_sorted,

            country_clean || '|' ||
            trim(
                regexp_replace(address_clean, '\\d+', '', 'g')
            ) AS address_no_number

        FROM s1
    ),

    -------------------------------------------------------
    -- SOURCE 2 KEYS
    -------------------------------------------------------

    s2_keys AS (
        SELECT
            entity_id AS candidate_entity_id,
            country_clean,

            name_clean,

            address_clean,

            country_clean || '|' ||
            left(
                regexp_replace(name_clean, '\\s+', '', 'g'),
                6
            ) AS name_prefix,

            country_clean || '|' ||
            coalesce(
                regexp_extract(address_clean, '(\\d+)', 1),
                ''
            ) AS address_number,

            country_clean || '|' ||
            array_to_string(
                list_sort(
                    string_split(name_clean, ' ')
                ),
                ''
            ) AS name_sorted,

            country_clean || '|' ||
            trim(
                regexp_replace(address_clean, '\\d+', '', 'g')
            ) AS address_no_number

        FROM s2
    ),

    -------------------------------------------------------
    -- SOURCE 3 KEYS
    -------------------------------------------------------

    s3_keys AS (
        SELECT
            entity_id AS candidate_entity_id,
            country_clean,

            name_clean,

            address_clean,

            country_clean || '|' ||
            left(
                regexp_replace(name_clean, '\\s+', '', 'g'),
                6
            ) AS name_prefix,

            country_clean || '|' ||
            coalesce(
                regexp_extract(address_clean, '(\\d+)', 1),
                ''
            ) AS address_number,

            country_clean || '|' ||
            array_to_string(
                list_sort(
                    string_split(name_clean, ' ')
                ),
                ''
            ) AS name_sorted,

            country_clean || '|' ||
            trim(
                regexp_replace(address_clean, '\\d+', '', 'g')
            ) AS address_no_number

        FROM s3
    ),

    -------------------------------------------------------
    -- SOURCE 2 NAME BLOCK
    -------------------------------------------------------

    s2_name AS (
        SELECT name_clean, country_clean, candidate_entity_id
        FROM s2_keys
        WHERE name_clean <> ''
        QUALIFY count(*) OVER (PARTITION BY country_clean, name_clean) <= 100
    ),

    s3_name AS (
        SELECT name_clean, country_clean, candidate_entity_id
        FROM s3_keys
        WHERE name_clean <> ''
        QUALIFY count(*) OVER (PARTITION BY country_clean, name_clean) <= 100
    ),

    s2_address AS (
        SELECT address_clean, country_clean, candidate_entity_id
        FROM s2_keys
        WHERE address_clean <> ''
        QUALIFY count(*) OVER (PARTITION BY country_clean, address_clean) <= 100
    ),

    s3_address AS (
        SELECT address_clean, country_clean, candidate_entity_id
        FROM s3_keys
        WHERE address_clean <> ''
        QUALIFY count(*) OVER (PARTITION BY country_clean, address_clean) <= 100
    ),

    s2_prefix AS (
        SELECT name_prefix, candidate_entity_id
        FROM s2_keys
        WHERE length(split_part(name_prefix, '|', 2)) >= 4
        QUALIFY count(*) OVER (PARTITION BY name_prefix) <= 50
    ),

    s3_prefix AS (
        SELECT name_prefix, candidate_entity_id
        FROM s3_keys
        WHERE length(split_part(name_prefix, '|', 2)) >= 4
        QUALIFY count(*) OVER (PARTITION BY name_prefix) <= 50
    ),

    s2_number AS (
        SELECT address_number, candidate_entity_id
        FROM s2_keys
        WHERE length(split_part(address_number, '|', 2)) >= 2
        QUALIFY count(*) OVER (PARTITION BY address_number) <= 50
    ),

    s3_number AS (
        SELECT address_number, candidate_entity_id
        FROM s3_keys
        WHERE length(split_part(address_number, '|', 2)) >= 2
        QUALIFY count(*) OVER (PARTITION BY address_number) <= 50
    ),

    -------------------------------------------------------
    -- NEW: SORTED NAME TOKEN BLOCK
    -- Catches word-order swaps (e.g. "Fils Grain" vs "Grain Fils")
    -------------------------------------------------------

    s2_name_sorted AS (
        SELECT name_sorted, candidate_entity_id
        FROM s2_keys
        WHERE length(split_part(name_sorted, '|', 2)) >= 4
        QUALIFY count(*) OVER (PARTITION BY name_sorted) <= 50
    ),

    s3_name_sorted AS (
        SELECT name_sorted, candidate_entity_id
        FROM s3_keys
        WHERE length(split_part(name_sorted, '|', 2)) >= 4
        QUALIFY count(*) OVER (PARTITION BY name_sorted) <= 50
    ),

    -------------------------------------------------------
    -- NEW: ADDRESS WITHOUT HOUSE NUMBER BLOCK
    -- Catches reformatted / missing house numbers
    -------------------------------------------------------

    s2_street AS (
        SELECT address_no_number, candidate_entity_id
        FROM s2_keys
        WHERE length(split_part(address_no_number, '|', 2)) >= 6
        QUALIFY count(*) OVER (PARTITION BY address_no_number) <= 50
    ),

    s3_street AS (
        SELECT address_no_number, candidate_entity_id
        FROM s3_keys
        WHERE length(split_part(address_no_number, '|', 2)) >= 6
        QUALIFY count(*) OVER (PARTITION BY address_no_number) <= 50
    ),

    -------------------------------------------------------
    -- ALL BLOCKS
    -------------------------------------------------------

    candidates AS (

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_name s2
          ON s1.name_clean = s2.name_clean AND s1.country_clean = s2.country_clean

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_name s3
          ON s1.name_clean = s3.name_clean AND s1.country_clean = s3.country_clean

        UNION

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_address s2
          ON s1.address_clean = s2.address_clean AND s1.country_clean = s2.country_clean

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_address s3
          ON s1.address_clean = s3.address_clean AND s1.country_clean = s3.country_clean

        UNION

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_prefix s2
          ON s1.name_prefix = s2.name_prefix

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_prefix s3
          ON s1.name_prefix = s3.name_prefix

        UNION

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_number s2
          ON s1.address_number = s2.address_number
         AND s1.country_clean = split_part(s2.address_number, '|', 1)

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_number s3
          ON s1.address_number = s3.address_number
         AND s1.country_clean = split_part(s3.address_number, '|', 1)

        UNION

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_name_sorted s2
          ON s1.name_sorted = s2.name_sorted

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_name_sorted s3
          ON s1.name_sorted = s3.name_sorted

        UNION

        SELECT s1.source1_entity_id, s2.candidate_entity_id
        FROM s1_keys s1 JOIN s2_street s2
          ON s1.address_no_number = s2.address_no_number

        UNION

        SELECT s1.source1_entity_id, s3.candidate_entity_id
        FROM s1_keys s1 JOIN s3_street s3
          ON s1.address_no_number = s3.address_no_number
    )

    SELECT DISTINCT
        source1_entity_id,
        candidate_entity_id
    FROM candidates
    """

    return query
# =========================================================
# TRAINING CANDIDATES
# =========================================================

def generate_training_candidates(con):

    print("Creating training candidate pairs...")

    query = generate_blocked_pairs(con)

    # Save blocked candidates as a temporary DuckDB table.
    con.execute(
        "CREATE OR REPLACE TEMP TABLE blocked_candidates AS " + query
    )

    blocked_count = con.execute(
        "SELECT COUNT(*) FROM blocked_candidates"
    ).fetchone()[0]

    print(f"Blocked candidate pairs: {blocked_count:,}")

    # -----------------------------------------------------
    # ADD ALL TRUE MATCHES
    #
    # This guarantees that training positives are not lost
    # because of blocking.
    # -----------------------------------------------------

    print("Adding ground-truth positive pairs...")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE training_candidates AS

        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM blocked_candidates

        UNION

        SELECT
            gt.source1_entity_id,
            trim(match_id) AS candidate_entity_id

        FROM read_csv(
            ?,
            sep='\t',
            header=true,
            all_varchar=true
        ) gt,

        LATERAL unnest(
            string_split(
                coalesce(gt.matched_entity_ids, ''),
                ','
            )
        ) AS matches(match_id)

        WHERE trim(match_id) <> ''
          AND (
                trim(match_id) LIKE 'S2-%'
                OR trim(match_id) LIKE 'S3-%'
          )
    """, [str(TRAIN_DIR / "train_ground_truth.tsv")])

    total = con.execute(
        "SELECT COUNT(*) FROM training_candidates"
    ).fetchone()[0]

    positives = con.execute("""
        SELECT COUNT(*)
        FROM training_candidates c
        JOIN read_csv(
            ?,
            sep='\t',
            header=true,
            all_varchar=true
        ) gt
        ON c.source1_entity_id = gt.source1_entity_id
        WHERE ',' || coalesce(gt.matched_entity_ids, '') || ','
              LIKE '%,' || c.candidate_entity_id || ',%'
    """, [str(TRAIN_DIR / "train_ground_truth.tsv")]).fetchone()[0]

    print(f"Training candidate pairs: {total:,}")
    print(f"Positive pairs included: {positives:,}")

    output_path = OUTPUT_DIR / "training_candidate_pairs.tsv"

    print()
    print("Writing training candidates...")

    con.execute("""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM training_candidates
            ORDER BY source1_entity_id, candidate_entity_id
        )
        TO ?
        WITH (
            FORMAT CSV,
            DELIMITER '\t',
            HEADER TRUE
        )
    """, [str(output_path)])

    print(f"Saved: {output_path}")

    return total


# =========================================================
# TEST CANDIDATES
# =========================================================
def generate_test_candidates(con):

    print()
    print("Creating final test candidate set...")

    query = generate_blocked_pairs(con)

    con.execute(
        "CREATE OR REPLACE TEMP TABLE test_candidates AS " + query
    )

    pair_count = con.execute(
        "SELECT COUNT(*) FROM test_candidates"
    ).fetchone()[0]

    print(f"Test candidate pairs: {pair_count:,}")

    # -----------------------------------------------------
    # IMPORTANT:
    # Do NOT use string_agg over 35M rows.
    #
    # Instead:
    # 1. Write sorted raw candidate pairs to disk.
    # 2. Stream the file row-by-row.
    # 3. Build one output row per Source 1 entity.
    # -----------------------------------------------------

    raw_path = OUTPUT_DIR / "test_candidates_raw.tsv"
    output_path = OUTPUT_DIR / "candidate_pairs.tsv"

    print()
    print("Writing raw candidate pairs to disk...")
    print("This avoids the memory-heavy string_agg operation.")

    con.execute("""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM test_candidates
            ORDER BY
                source1_entity_id,
                candidate_entity_id
        )
        TO ?
        WITH (
            FORMAT CSV,
            DELIMITER '\t',
            HEADER TRUE
        )
    """, [str(raw_path)])

    print("Raw candidate file written.")
    print("Creating final candidate_pairs.tsv...")

    import csv

    # -----------------------------------------------------
    # Create lookup for ALL Source 1 IDs.
    # This guarantees every Source 1 entity gets a row.
    # -----------------------------------------------------

    s1_ids = con.execute(
        "SELECT entity_id FROM s1 ORDER BY entity_id"
    ).fetchall()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as out_file:

        writer = csv.writer(
            out_file,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])

        current_id = None
        current_candidates = []

        s1_index = 0

        def write_empty_until(target_id):

            nonlocal s1_index

            while (
                s1_index < len(s1_ids)
                and s1_ids[s1_index][0] < target_id
            ):
                writer.writerow([
                    s1_ids[s1_index][0],
                    ""
                ])

                s1_index += 1

        with open(
            raw_path,
            "r",
            newline="",
            encoding="utf-8"
        ) as raw_file:

            reader = csv.DictReader(
                raw_file,
                delimiter="\t"
            )

            for row in reader:

                source1_id = row["source1_entity_id"]
                candidate_id = row["candidate_entity_id"]

                # Handle S1 entities with no candidates.
                if current_id is None:
                    write_empty_until(source1_id)
                    current_id = source1_id

                elif source1_id != current_id:

                    # Write previous group.
                    writer.writerow([
                        current_id,
                        ",".join(current_candidates)
                    ])

                    s1_index += 1

                    # Fill any S1 IDs between groups.
                    write_empty_until(source1_id)

                    current_id = source1_id
                    current_candidates = []

                current_candidates.append(candidate_id)

            # Write final candidate group.
            if current_id is not None:

                writer.writerow([
                    current_id,
                    ",".join(current_candidates)
                ])

                s1_index += 1

        # -------------------------------------------------
        # Write remaining Source 1 entities with no matches.
        # -------------------------------------------------

        while s1_index < len(s1_ids):

            writer.writerow([
                s1_ids[s1_index][0],
                ""
            ])

            s1_index += 1

    print()
    print("Final candidate_pairs.tsv created.")

    print(f"Source 1 entities: {len(s1_ids):,}")
    print(f"Total candidate pairs: {pair_count:,}")
    print(f"Saved: {output_path}")

    # -----------------------------------------------------
    # Raw file is no longer needed.
    # -----------------------------------------------------

    try:
        raw_path.unlink()
        print("Temporary raw candidate file removed.")
    except Exception:
        print(
            f"Temporary file retained at: {raw_path}"
        )

    return pair_count
# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AMAZON BUSINESS ENTITY RESOLUTION")
    print("SCALABLE DUCKDB BLOCKING")
    print("=" * 60)

    con = create_connection()

    try:

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        create_views(
            con,
            TRAIN_DIR / "train_source1.tsv",
            TRAIN_DIR / "train_source2.tsv",
            TRAIN_DIR / "train_source3.tsv"
        )

        generate_training_candidates(con)

        # -------------------------------------------------
        # TEST
        # -------------------------------------------------

        print()
        print("=" * 60)
        print("NOW PROCESSING TEST DATA")
        print("=" * 60)

        create_views(
            con,
            TEST_DIR / "test_source1.tsv",
            TEST_DIR / "test_source2.tsv",
            TEST_DIR / "test_source3.tsv"
        )

        generate_test_candidates(con)

        print()
        print("=" * 60)
        print("BLOCKING FINISHED SUCCESSFULLY")
        print("=" * 60)

    finally:
        con.close()