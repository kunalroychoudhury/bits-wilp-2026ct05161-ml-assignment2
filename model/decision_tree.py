from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = "Decision Tree"


def create_model() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=42)
