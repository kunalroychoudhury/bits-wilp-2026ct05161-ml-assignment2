from sklearn.neighbors import KNeighborsClassifier

MODEL_NAME = "kNN"


def create_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=5)
