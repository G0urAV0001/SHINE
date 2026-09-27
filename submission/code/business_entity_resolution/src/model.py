from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "entity_matcher.pkl"


def create_model():
    """
    Create the entity matching classification model.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return model


def train_model(X_train, y_train):
    """
    Train the entity matching model.
    """

    model = create_model()

    model.fit(X_train, y_train)

    return model


def predict_probability(model, X):
    """
    Return probability that each pair is a match.
    """

    probabilities = model.predict_proba(X)

    # Probability of class 1 (MATCH)
    return probabilities[:, 1]


def save_model(model):
    """
    Save the trained model.
    """

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


def load_model():
    """
    Load a previously trained model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    print("Entity Matching Model")
    print("---------------------")
    print("Model type: Random Forest")
    print(f"Model will be saved to: {MODEL_PATH}")