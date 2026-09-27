import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data"


def load_training_data():
    """
    Load the three training source files
    and the training ground truth.
    """

    source1 = pd.read_csv(
        DATA_DIR / "train" / "train_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        DATA_DIR / "train" / "train_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        DATA_DIR / "train" / "train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        DATA_DIR / "train" / "train_ground_truth.tsv",
        sep="\t"
    )

    return source1, source2, source3, ground_truth


def load_test_data():
    """
    Load the three test source files.
    """

    source1 = pd.read_csv(
        DATA_DIR / "test" / "test_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        DATA_DIR / "test" / "test_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        DATA_DIR / "test" / "test_source3.tsv",
        sep="\t"
    )

    return source1, source2, source3


def validate_columns(df, required_columns, source_name):
    """
    Check whether the expected columns exist.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing columns: "
            f"{missing_columns}"
        )


if __name__ == "__main__":

    print("Loading training data...\n")

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    print("Source 1 shape:", source1.shape)
    print("Source 2 shape:", source2.shape)
    print("Source 3 shape:", source3.shape)
    print("Ground truth shape:", ground_truth.shape)

    print("\nSource 1 columns:")
    print(source1.columns.tolist())

    print("\nSource 2 columns:")
    print(source2.columns.tolist())

    print("\nSource 3 columns:")
    print(source3.columns.tolist())

    print("\nGround truth columns:")
    print(ground_truth.columns.tolist())

    required_source_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    validate_columns(
        source1,
        required_source_columns,
        "Source 1"
    )

    validate_columns(
        source2,
        required_source_columns,
        "Source 2"
    )

    validate_columns(
        source3,
        required_source_columns,
        "Source 3"
    )

    required_ground_truth_columns = [
        "source1_entity_id",
        "matched_entity_ids"
    ]

    validate_columns(
        ground_truth,
        required_ground_truth_columns,
        "Ground Truth"
    )

    print("\n✓ All required columns are present.")
    print("✓ Training data loaded successfully.")