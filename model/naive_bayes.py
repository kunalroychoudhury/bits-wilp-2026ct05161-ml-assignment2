from sklearn.naive_bayes import GaussianNB

MODEL_NAME = "Naive Bayes"


def create_model() -> GaussianNB:
    return GaussianNB()
