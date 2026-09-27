from similarity import (
    name_similarity,
    address_similarity,
    country_similarity
)


def detect_mismatches(source1_row, candidate_row):
    """
    Detect important differences between two business records.

    Returns a dictionary containing mismatch information.
    """

    mismatches = []

    # -------------------------
    # Business name
    # -------------------------

    name_score = name_similarity(
        source1_row["business_name_clean"],
        candidate_row["business_name_clean"]
    )

    if name_score < 0.50:
        mismatches.append({
            "field": "business_name",
            "similarity": round(name_score, 4),
            "reason": "Business names are significantly different"
        })

    # -------------------------
    # Business address
    # -------------------------

    address_score = address_similarity(
        source1_row["business_address_clean"],
        candidate_row["business_address_clean"]
    )

    if address_score < 0.50:
        mismatches.append({
            "field": "business_address",
            "similarity": round(address_score, 4),
            "reason": "Business addresses are significantly different"
        })

    # -------------------------
    # Country
    # -------------------------

    country_score = country_similarity(
        source1_row["country_clean"],
        candidate_row["country_clean"]
    )

    if country_score == 0:
        mismatches.append({
            "field": "country",
            "similarity": 0.0,
            "reason": "Country values do not match"
        })

    return {
        "has_mismatch": len(mismatches) > 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches
    }


def get_mismatch_count(source1_row, candidate_row):
    """
    Return only the number of detected mismatches.
    """

    result = detect_mismatches(
        source1_row,
        candidate_row
    )

    return result["mismatch_count"]


if __name__ == "__main__":
    print("Mismatch detector loaded successfully.")