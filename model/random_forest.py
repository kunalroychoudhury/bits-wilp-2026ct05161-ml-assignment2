from sklearn.ensemble import RandomForestClassifier

MODEL_NAME = "Random Forest"


def create_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
