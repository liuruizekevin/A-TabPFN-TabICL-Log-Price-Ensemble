"""Row-local feature definitions for the domain TabPFN component."""

import numpy as np
import pandas as pd


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

    Existing engineered columns are rebuilt to make the transformation repeatable.
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
