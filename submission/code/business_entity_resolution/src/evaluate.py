import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall and F0.5 score.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f05
    }


def find_best_threshold(
    y_true,
    probabilities,
    start=0.50,
    end=0.95,
    step=0.01
):
    """
    Find the threshold that gives the highest F0.5
    on validation data.
    """

    best_threshold = start
    best_score = -1

    threshold = start

    while threshold <= end:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

        threshold += step

    return best_threshold, best_score


def evaluate_model(model, X_validation, y_validation):
    """
    Evaluate the model on validation data
    and find the best F0.5 threshold.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold, best_f05 = find_best_threshold(
        y_validation,
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_validation,
        predictions
    )

    return {
        "threshold": threshold,
        "best_f0.5": best_f05,
        "precision": metrics["precision"],
        "recall": metrics["recall"]
    }


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")