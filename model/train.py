"""Train and evaluate classification model plugins on the processed split."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.preprocess import CLASS_COLUMN, PROJECT_ROOT

DEFAULT_PLUGINS = (
    "model.logistic_regression",
    "model.decision_tree",
    "model.knn",
    "model.naive_bayes",
    "model.random_forest",
)


def load_split(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if CLASS_COLUMN not in train.columns or CLASS_COLUMN not in test.columns:
        raise ValueError(f"Both CSV files must contain {CLASS_COLUMN}.")
    if train.empty or test.empty:
        raise ValueError("Both training and test datasets must contain rows.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Training and test datasets must not contain missing values.")

    x_train = train.drop(columns=CLASS_COLUMN)
    x_test = test.drop(columns=CLASS_COLUMN)
    if x_train.empty or x_train.columns.tolist() != x_test.columns.tolist():
        raise ValueError("Training and test feature columns must match in order.")
    y_train = train[CLASS_COLUMN]
    y_test = test[CLASS_COLUMN]
    if set(y_train) != set(y_test) or y_train.nunique() < 3:
        raise ValueError("Both datasets must contain the same three or more classes.")
    return x_train, x_test, y_train, y_test


def run_models(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    plugin_names: tuple[str, ...] = DEFAULT_PLUGINS,
) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = load_split(train_path, test_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    seen_names = set()

    for plugin_name in plugin_names:
        plugin = importlib.import_module(plugin_name)
        name = plugin.MODEL_NAME
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise ValueError(f"Plugin {plugin_name} has an invalid or duplicate MODEL_NAME.")
        seen_names.add(name)
        estimator = plugin.create_model()
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        probabilities = estimator.predict_proba(x_test)
        unpredicted_classes = sorted(set(y_test) - set(predicted))
        if unpredicted_classes:
            warnings.warn(
                f"{name} predicted no examples of {unpredicted_classes}; "
                "precision for these classes is set to zero.",
                UserWarning,
            )
        missed_classes = [
            label
            for label in estimator.classes_
            if not ((y_test.to_numpy() == label) & (predicted == label)).any()
        ]
        if missed_classes:
            warnings.warn(
                f"{name} correctly identified no test examples of {missed_classes}; "
                "recall for these classes is zero.",
                UserWarning,
            )
        metrics = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "AUC": roc_auc_score(
                y_test,
                probabilities,
                labels=estimator.classes_,
                multi_class="ovr",
                average="macro",
            ),
            "Precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "Recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "F1": f1_score(y_test, predicted, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predicted),
        }
        results.append(metrics)
        model_path = model_dir / f"{plugin_name.replace('.', '_')}.joblib"
        joblib.dump(estimator, model_path)
        print(f"{name}: {estimator}")
        print(f"  Saved model: {model_path}")

    if not results:
        raise ValueError("At least one model plugin is required.")
    comparison = pd.DataFrame(results)
    output_path = output_dir / "model_metrics.csv"
    comparison.to_csv(output_path, index=False, float_format="%.17g", lineterminator="\n")
    print("\nHeld-out test metrics (macro average; AUC one-vs-rest):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics: {output_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "train_data.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "test_data.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "model" / "results")
    parser.add_argument(
        "--plugins",
        nargs="+",
        default=list(DEFAULT_PLUGINS),
        help="Importable module names defining MODEL_NAME and create_model().",
    )
    arguments = parser.parse_args()
    run_models(arguments.train, arguments.test, arguments.output_dir, tuple(arguments.plugins))


if __name__ == "__main__":
    main()
