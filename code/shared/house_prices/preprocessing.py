"""Deterministic feature definitions and training-fold-only preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_GROUPS = {
    "area": ["TotalSF", "TotalFinishedSF"],
    "age": ["HouseAge", "RemodelAge", "GarageAge", "WasRemodeled"],
    "baths": ["TotalBath"],
    "porch": ["TotalPorchSF"],
    "quality": ["QualityLivingArea", "QualityTotalSF"],
}
ALL_GROUPS = tuple(FEATURE_GROUPS)
ENGINEERED_COLUMNS = sum(FEATURE_GROUPS.values(), [])


def read_data(path):
    """Preserve the literal category 'None'; only NA and empty cells are missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=["NA", ""])


def deterministic_features(X, semantic=False, groups=ALL_GROUPS):
    """Apply row-local rules only; no statistics, targets, or test-set fitting.

    Existing engineered columns are rebuilt so exported A-stage data is a valid
    input to every experiment, including experiments with no engineered groups.
    Missing component sizes are temporarily zero in derived sums; their original
    columns retain missing values for later fold-fitted imputation.
    """
    if "SalePrice" in X:
        raise ValueError("SalePrice must be separated before preprocessing.")
    out = X.drop(columns=ENGINEERED_COLUMNS, errors="ignore").copy()
    if "MSSubClass" in out:
        out["MSSubClass"] = out["MSSubClass"].map(
            lambda v: str(int(float(v))) if pd.notna(v) else np.nan
        ).astype(object)
    # pandas 3 string columns become ordinary object columns for sklearn.
    for c in out.select_dtypes(include=["string", "object"]).columns:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)

    def num(c):
        return pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if semantic:
        for c in ["Alley", "Fence"]:
            out[c] = out[c].fillna("None")
        for columns, presence in [
            (["PoolQC"], num("PoolArea") > 0),
            (["FireplaceQu"], num("Fireplaces") > 0),
            (["MiscFeature"], num("MiscVal") > 0),
            (["GarageType", "GarageFinish", "GarageQual", "GarageCond"], num("GarageCars") > 0),
            (["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], num("TotalBsmtSF") > 0),
            (["MasVnrType"], num("MasVnrArea") > 0),
        ]:
            for c in columns:
                out[c] = out[c].fillna(pd.Series(np.where(presence, "Unknown", "None"), index=out.index))
        # Size missingness remains available as an explicit indicator.
        for c in ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]:
            out[c + "Missing"] = out[c].isna().astype(int)
            out[c] = out[c].fillna(0.0)
    total_sf = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
    if "area" in groups:
        out["TotalSF"] = total_sf
        out["TotalFinishedSF"] = num("GrLivArea") + num("BsmtFinSF1") + num("BsmtFinSF2")
    if "age" in groups:
        out["HouseAge"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
        out["RemodelAge"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)
        out["GarageAge"] = (num("YrSold") - num("GarageYrBlt")).clip(lower=0).where(num("GarageYrBlt") > 0, 0)
        out["WasRemodeled"] = (num("YearRemodAdd") != num("YearBuilt")).astype(int)
    if "baths" in groups:
        out["TotalBath"] = num("FullBath") + 0.5 * num("HalfBath") + num("BsmtFullBath") + 0.5 * num("BsmtHalfBath")
    if "porch" in groups:
        out["TotalPorchSF"] = sum(num(c) for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"])
    if "quality" in groups:
        out["QualityLivingArea"] = num("OverallQual") * num("GrLivArea")
        out["QualityTotalSF"] = num("OverallQual") * total_sf
    return out


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def __init__(self, semantic=False, groups=()):
        self.semantic = semantic
        self.groups = groups

    def fit(self, X, y=None):
        self.fit_row_ids_ = X["Id"].tolist() if "Id" in X else X.index.tolist()
        self.n_fit_rows_ = len(X)
        return self

    def transform(self, X):
        # Id identifies rows; it is never a predictor.
        return deterministic_features(X, self.semantic, self.groups).drop(columns="Id", errors="ignore")


class SkewedLogTransformer(TransformerMixin, BaseEstimator):
    """Select continuous nonnegative columns by TRAINING-FOLD skewness only."""
    def __init__(self, enabled=False, threshold=0.75):
        self.enabled = enabled
        self.threshold = threshold

    def fit(self, X, y=None):
        values = pd.DataFrame(X)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(values.columns, dtype=object)
        self.skewness_ = values.skew().fillna(0).to_numpy()
        self.selected_indices_ = [i for i, c in enumerate(values) if self.enabled and values[c].nunique() >= 10 and values[c].min() >= 0 and self.skewness_[i] > self.threshold]
        self.selected_columns_ = [str(values.columns[i]) for i in self.selected_indices_]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        if self.selected_indices_:
            values[:, self.selected_indices_] = np.log1p(np.maximum(values[:, self.selected_indices_], 0))
        return values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


def make_preprocessor(skew=False):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
        ("skew", SkewedLogTransformer(enabled=skew)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ], remainder="drop", verbose_feature_names_out=True)
