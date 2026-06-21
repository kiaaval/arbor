import csv
import warnings

from schema import feature


def load_data(path: str, target_column: str, feature_columns: list[str] | None):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if feature_columns is None:
        feature_columns = [k for k in rows[0].keys() if k != target_column]
        if "#" in feature_columns:
            warnings.warn(
                "Column '#' looks like an identifier and will be used as a "
                'feature; it will dominate the split. Pass --features "[Example]Weather, Temperature" to'
                "control which columns are used as features.",
                stacklevel=2,
            )

    features: list[feature] = [
        {"name": col, "array": [row[col] for row in rows]} for col in feature_columns
    ]
    labels = [row[target_column] for row in rows]
    return features, labels
