def generate_evidence(source1_row, candidate_row, features, threshold):
    """
    Generate human-readable evidence for an entity match.
    """

    evidence = []
    mismatches = []

    name_score = features["name_similarity"]
    address_score = features["address_similarity"]
    country_score = features["country_similarity"]

    # Name evidence
    if name_score >= 0.85:
        evidence.append(
            f"Business names are highly similar ({name_score:.2f})"
        )
    elif name_score >= 0.65:
        evidence.append(
            f"Business names are moderately similar ({name_score:.2f})"
        )
    else:
        mismatches.append(
            f"Business names have low similarity ({name_score:.2f})"
        )

    # Address evidence
    if address_score >= 0.85:
        evidence.append(
            f"Addresses are highly similar ({address_score:.2f})"
        )
    elif address_score >= 0.65:
        evidence.append(
            f"Addresses are moderately similar ({address_score:.2f})"
        )
    else:
        mismatches.append(
            f"Addresses have low similarity ({address_score:.2f})"
        )

    # Country evidence
    if country_score == 1.0:
        evidence.append("Country values agree")
    else:
        mismatches.append("Country values do not agree")

    # Final decision
    combined_score = (
        0.45 * name_score
        + 0.45 * address_score
        + 0.10 * country_score
    )

    decision = (
        "MATCH"
        if combined_score >= threshold
        else "NO_MATCH"
    )

    return {
        "decision": decision,
        "score": round(combined_score, 4),
        "evidence": evidence,
        "mismatches": mismatches
    }


if __name__ == "__main__":

    example_features = {
        "name_similarity": 0.92,
        "address_similarity": 0.88,
        "country_similarity": 1.0
    }

    result = generate_evidence(
        None,
        None,
        example_features,
        threshold=0.80
    )

    print("Decision:", result["decision"])
    print("Score:", result["score"])

    print("\nEvidence:")
    for item in result["evidence"]:
        print("✓", item)

    print("\nMismatches:")
    for item in result["mismatches"]:
        print("⚠", item)