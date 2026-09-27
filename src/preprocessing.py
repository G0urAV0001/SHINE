import pandas as pd
import re


# Common business-name abbreviations
NAME_REPLACEMENTS = {
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "limited": "ltd",
    "ltd.": "ltd",
    "corporation": "corp",
    "corp.": "corp",
    "incorporated": "inc",
    "inc.": "inc",
    "company": "co",
    "co.": "co",
}


# Common address abbreviations
ADDRESS_REPLACEMENTS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "lane": "ln",
    "drive": "dr",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
}


def normalize_basic(value):
    """
    Basic normalization shared by names, addresses and countries.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Replace punctuation with spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_business_name(name):
    """
    Normalize a business name.
    """

    name = normalize_basic(name)

    if not name:
        return ""

    for old, new in NAME_REPLACEMENTS.items():
        name = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            name
        )

    return re.sub(r"\s+", " ", name).strip()


def normalize_address(address):
    """
    Normalize a business address.
    """

    address = normalize_basic(address)

    if not address:
        return ""

    for old, new in ADDRESS_REPLACEMENTS.items():
        address = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            address
        )

    return re.sub(r"\s+", " ", address).strip()


def normalize_country(country):
    """
    Normalize country labels.

    IMPORTANT:
    We do not restrict countries to US/India.
    This allows unseen countries such as France.
    """

    return normalize_basic(country)


def preprocess_dataframe(df):
    """
    Add normalized columns while preserving
    the original columns.
    """

    df = df.copy()

    df["business_name_clean"] = (
        df["business_name"]
        .map(normalize_business_name)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["country_clean"] = (
        df["country"]
        .map(normalize_country)
    )

    return df


def preprocess_all_sources(source1, source2, source3):
    """
    Preprocess all three sources.
    """

    source1 = preprocess_dataframe(source1)
    source2 = preprocess_dataframe(source2)
    source3 = preprocess_dataframe(source3)

    return source1, source2, source3


if __name__ == "__main__":

    # Small test only.
    # We deliberately do NOT load the millions of records here.

    test_data = pd.DataFrame({
        "entity_id": ["S1-TEST"],
        "business_name": [
            "ABC Technologies Private Limited."
        ],
        "business_address": [
            "123 MG Road, Bangalore"
        ],
        "country": ["India"]
    })

    result = preprocess_dataframe(test_data)

    print(result.to_string(index=False))