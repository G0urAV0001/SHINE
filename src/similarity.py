import re
from difflib import SequenceMatcher


def normalize_text(text):
    """
    Normalize text before comparison.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def exact_similarity(value1, value2):
    """
    Compare two values after normalization.
    Returns 1.0 for an exact match and 0.0 otherwise.
    """

    value1 = normalize_text(value1)
    value2 = normalize_text(value2)

    if not value1 or not value2:
        return 0.0

    return 1.0 if value1 == value2 else 0.0


def sequence_similarity(value1, value2):
    """
    Character-level similarity using SequenceMatcher.
    Returns a value between 0 and 1.
    """

    value1 = normalize_text(value1)
    value2 = normalize_text(value2)

    if not value1 or not value2:
        return 0.0

    return SequenceMatcher(None, value1, value2).ratio()


def token_similarity(value1, value2):
    """
    Jaccard-style token similarity.
    """

    value1 = normalize_text(value1)
    value2 = normalize_text(value2)

    if not value1 or not value2:
        return 0.0

    tokens1 = set(value1.split())
    tokens2 = set(value2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)


def name_similarity(name1, name2):
    """
    Calculate business-name similarity.
    """

    sequence_score = sequence_similarity(name1, name2)
    token_score = token_similarity(name1, name2)

    return (0.6 * sequence_score) + (0.4 * token_score)


def address_similarity(address1, address2):
    """
    Calculate business-address similarity.
    """

    sequence_score = sequence_similarity(address1, address2)
    token_score = token_similarity(address1, address2)

    return (0.4 * sequence_score) + (0.6 * token_score)


def country_similarity(country1, country2):
    """
    Compare country values.
    """

    return exact_similarity(country1, country2)


if __name__ == "__main__":

    name1 = "ABC Technologies Pvt Ltd"
    name2 = "ABC Technology Private Limited"

    address1 = "MG Road, Bangalore"
    address2 = "M G Road Bangalore"

    country1 = "India"
    country2 = "India"

    print("Name similarity:",
          name_similarity(name1, name2))

    print("Address similarity:",
          address_similarity(address1, address2))

    print("Country similarity:",
          country_similarity(country1, country2))