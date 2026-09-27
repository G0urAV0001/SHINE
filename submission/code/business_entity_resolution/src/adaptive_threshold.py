import numpy as np


def calculate_threshold(scores, percentile=95):
    """
    Calculate an adaptive threshold from model scores.

    Parameters
    ----------
    scores : list or array
        Matching scores produced by the ML model.

    percentile : float
        Percentile used to determine the threshold.

    Returns
    -------
    float
        Calculated threshold.
    """

    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return 0.5

    threshold = np.percentile(scores, percentile)

    return float(threshold)


def classify_score(score, threshold):
    """
    Classify a pair using the calculated threshold.

    Returns:
        MATCH or NO_MATCH
    """

    if score >= threshold:
        return "MATCH"

    return "NO_MATCH"


def adaptive_match(score, threshold):
    """
    Return True if the score passes the threshold.
    """

    return score >= threshold


if __name__ == "__main__":

    # Example model scores
    example_scores = [
        0.91,
        0.87,
        0.95,
        0.72,
        0.89,
        0.43,
        0.96,
        0.67,
        0.93,
        0.81
    ]

    threshold = calculate_threshold(example_scores)

    print("Adaptive Threshold:", threshold)

    test_score = 0.94

    result = classify_score(test_score, threshold)

    print("Test Score:", test_score)
    print("Decision:", result)

