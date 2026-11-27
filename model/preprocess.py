import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "garments_worker_productivity.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT
RANDOM_SEED = 42
TEST_SIZE = 0.20

DATE_COLUMN = "date"
TARGET_COLUMN = "actual_productivity"
CLASS_COLUMN = "productivity_class"
NUMERIC_FEATURES = [
    "targeted_productivity",
    "smv",
    "wip",
    "over_time",
    "incentive",
    "idle_time",
    "idle_men",
    "no_of_style_change",
    "no_of_workers",
]
CATEGORICAL_FEATURES = ["quarter", "department", "day", "team"]
REQUIRED_COLUMNS = [
    DATE_COLUMN,
    "quarter",
    "department",
    "day",
    "team",
    *NUMERIC_FEATURES,
    TARGET_COLUMN,
]


def load_and_clean_data(input_path: Path) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(input_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN], format="%m/%d/%Y", errors="raise"
    )

    for column in ("quarter", "department", "day"):
        data[column] = data[column].astype("string").str.strip()

    team = pd.to_numeric(data["team"], errors="raise")
    if team.isna().any() or not team.mod(1).eq(0).all():
        raise ValueError("The team identifier must contain whole-number values.")
    data["team"] = team.astype("int64").astype("string")

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values.")

    duplicates_removed = int(data.duplicated().sum())
    data = data.drop_duplicates()
    data = data.sort_values(
        by=REQUIRED_COLUMNS, kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    return data, duplicates_removed


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def assign_classes(target: pd.Series) -> pd.Series:
    labels = np.select(
        [target < 0.40, target < 0.80],
        ["low", "medium"],
        default="high",
    )
    return pd.Series(labels, index=target.index, name=CLASS_COLUMN)


def write_processed_splits(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    data, duplicates_removed = load_and_clean_data(input_path)
    labels = assign_classes(data[TARGET_COLUMN])
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    row_indices = np.arange(len(data))

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    preprocessor = make_preprocessor()
    train_features = features.iloc[train_indices]
    test_features = features.iloc[test_indices]
    train_values = preprocessor.fit_transform(train_features)
    test_values = preprocessor.transform(test_features)
    output_columns = preprocessor.get_feature_names_out()

    train_output = pd.DataFrame(train_values, columns=output_columns)
    train_output.insert(0, CLASS_COLUMN, labels.iloc[train_indices].to_numpy())
    test_output = pd.DataFrame(test_values, columns=output_columns)
    test_output.insert(0, CLASS_COLUMN, labels.iloc[test_indices].to_numpy())

    if train_output.isna().any().any() or test_output.isna().any().any():
        raise ValueError("Preprocessing produced missing values.")
    if len(train_output) + len(test_output) != len(data):
        raise RuntimeError("The train/test split does not cover every input row.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    csv_options = {"index": False, "float_format": "%.17g", "lineterminator": "\n"}
    train_output.to_csv(train_path, **csv_options)
    test_output.to_csv(test_path, **csv_options)

    print(f"Exact duplicate rows removed: {duplicates_removed}")
    print(f"Input rows after duplicate removal: {len(data)}")
    print(f"Training rows: {len(train_output)}")
    print(f"Test rows: {len(test_output)}")
    print("Training class counts:")
    print(labels.iloc[train_indices].value_counts().sort_index().to_string())
    print("Test class counts:")
    print(labels.iloc[test_indices].value_counts().sort_index().to_string())
    print(f"Processed feature columns: {len(output_columns)}")
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the garment productivity dataset reproducibly."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    write_processed_splits(arguments.input, arguments.output_dir)


if __name__ == "__main__":
    main()
