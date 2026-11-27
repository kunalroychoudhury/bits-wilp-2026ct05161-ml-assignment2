from sklearn.linear_model import LogisticRegression

MODEL_NAME = "Logistic Regression"


def create_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)
