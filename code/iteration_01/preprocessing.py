"""Feature handoff: row-local engineering only; missing values remain unfitted."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from house_prices.preprocessing import ALL_GROUPS, deterministic_features, read_data


def export_features(data_dir=ROOT / "data/raw", output_dir=ROOT / "features"):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = read_data(data_dir / "train.csv"), read_data(data_dir / "test.csv")
    X_train = deterministic_features(train.drop(columns="SalePrice"), semantic=False, groups=ALL_GROUPS)
    X_test = deterministic_features(test, semantic=False, groups=ALL_GROUPS)
    assert X_train.columns.tolist() == X_test.columns.tolist()
    X_train["SalePrice"] = train["SalePrice"]
    X_train.to_csv(output_dir / "train_processed.csv", index=False)
    X_test.to_csv(output_dir / "test_processed.csv", index=False)
    return X_train, X_test


if __name__ == "__main__":
    export_features()
