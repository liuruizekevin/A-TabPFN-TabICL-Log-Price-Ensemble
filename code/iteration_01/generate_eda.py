"""Reproduce descriptive analysis and figures from the unmodified competition data.

Usage: python3 scripts/generate_eda.py
All training observations are retained. These full-training summaries are
descriptive and must not be used as a substitute for fold-local preprocessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


BLUE = "#28659A"
TEAL = "#21867A"
ORANGE = "#CC7722"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_raw(path: Path) -> pd.DataFrame:
    """Preserve the valid masonry category 'None'; only blank/NA are missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=["", "NA"])


def distribution(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std_sample": float(series.std(ddof=1)),
        "minimum": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "maximum": float(series.max()),
        "skewness_sample": float(series.skew()),
    }


def missing_rows(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for name in test.columns:
        tc, ec = int(train[name].isna().sum()), int(test[name].isna().sum())
        if tc or ec:
            rows.append({
                "feature": name,
                "train_count": tc, "train_percent": tc / len(train) * 100,
                "test_count": ec, "test_percent": ec / len(test) * 100,
            })
    return sorted(rows, key=lambda r: max(r["train_percent"], r["test_percent"]), reverse=True)


def save_figure(fig, directory: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def generate(data_dir: Path, output_dir: Path) -> dict:
    train, test = read_raw(data_dir / "train.csv"), read_raw(data_dir / "test.csv")
    assert train.shape == (1460, 81), train.shape
    assert test.shape == (1459, 80), test.shape
    assert train.drop(columns="SalePrice").columns.tolist() == test.columns.tolist()
    assert train["Id"].is_unique and test["Id"].is_unique
    assert (train["SalePrice"] > 0).all()
    assert "None" in set(train["MasVnrType"].dropna())
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.dpi": 110, "savefig.dpi": 220,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    price = train["SalePrice"]
    log_price = np.log(price)
    log1p_price = np.log1p(price)
    missing = missing_rows(train, test)
    numeric = train.drop(columns=["Id", "SalePrice"]).select_dtypes(include="number").columns.tolist()
    categorical = train.drop(columns=["Id", "SalePrice"]).select_dtypes(exclude="number").columns.tolist()
    corr_rows = []
    for feature in numeric:
        valid = train[feature].notna() & price.notna()
        corr_rows.append({
            "feature": feature, "paired_observations": int(valid.sum()),
            "pearson_with_sale_price": float(train.loc[valid, feature].corr(price[valid])),
            "pearson_with_ln_sale_price": float(train.loc[valid, feature].corr(log_price[valid])),
        })
    corr_rows.sort(key=lambda row: abs(row["pearson_with_ln_sale_price"]), reverse=True)
    diagnostic = train.loc[(train["GrLivArea"] > 4000) & (price < 300000),
                           ["Id", "GrLivArea", "SalePrice", "OverallQual"]]
    garage_future = test.loc[test["GarageYrBlt"] > test["YrSold"],
                             ["Id", "GarageYrBlt", "YrSold"]]
    amenity_flags = {
        category: test.loc[test[category].isna() & test[size].gt(0),
                           ["Id", size]].to_dict("records")
        for category, size in [("PoolQC", "PoolArea"), ("MiscFeature", "MiscVal")]
    }
    summary = {
        "schema_version": 1,
        "data_policy": {
            "raw_data_modified": False,
            "training_rows_retained": len(train),
            "training_rows_removed": 0,
            "test_target_available": False,
            "na_tokens": ["", "NA"],
            "literal_None_preserved": True,
            "scope": "Descriptive full-training EDA only; no target-driven feature selection or fold fitting is performed here.",
            "missingness_semantics": "NA can indicate an absent amenity; consult data_description.txt before imputation. It does not always mean an unknown value.",
        },
        "dataset": {
            "train_rows": len(train), "train_columns": train.shape[1],
            "test_rows": len(test), "test_columns": test.shape[1],
            "predictors_excluding_id": len(test.columns)-1,
            "raw_numeric_predictor_count": len(numeric),
            "raw_categorical_predictor_count": len(categorical),
            "raw_numeric_predictors": numeric,
            "raw_categorical_predictors": categorical,
            "numeric_code_note": "MSSubClass is numeric on disk but is a building-class code; MoSold is a calendar category.",
            "train_missing_cells": int(train.isna().sum().sum()),
            "test_missing_cells": int(test.isna().sum().sum()),
            "train_missing_feature_count": int(train.drop(columns="SalePrice").isna().any().sum()),
            "test_missing_feature_count": int(test.isna().any().sum()),
            "train_duplicate_ids": int(train["Id"].duplicated().sum()),
            "test_duplicate_ids": int(test["Id"].duplicated().sum()),
            "train_duplicate_entire_rows": int(train.duplicated().sum()),
            "test_duplicate_entire_rows": int(test.duplicated().sum()),
            "id_overlap": len(set(train["Id"]) & set(test["Id"])),
        },
        "target": {
            "sale_price_usd": distribution(price),
            "ln_sale_price": distribution(log_price),
            "log1p_sale_price": distribution(log1p_price),
            "max_absolute_log1p_minus_ln": float((log1p_price-log_price).max()),
        },
        "missingness": missing,
        "mas_vnr_type": {
            "train_literal_None_count": int(train["MasVnrType"].eq("None").sum()),
            "test_literal_None_count": int(test["MasVnrType"].eq("None").sum()),
            "train_unknown_missing_count": int(train["MasVnrType"].isna().sum()),
            "test_unknown_missing_count": int(test["MasVnrType"].isna().sum()),
        },
        "numeric_correlations": corr_rows,
        "overall_quality": [
            {"overall_qual": int(key), "count": int(len(group)),
             "median_sale_price": float(group["SalePrice"].median())}
            for key, group in train.groupby("OverallQual", sort=True)
        ],
        "diagnostic_flags": {
            "large_low_price_rule": "GrLivArea > 4000 sq ft and SalePrice < 300000 USD",
            "large_low_price_training_rows": diagnostic.to_dict("records"),
            "large_low_price_action": "Flagged for discussion; every observation is retained in EDA and no deletion is recommended from this plot alone.",
            "test_garage_year_later_than_sale": garage_future.to_dict("records"),
            "test_garage_year_action": "Raw values retained by EDA. A year after the recorded sale is a consistency flag, not proof of an error.",
            "test_missing_amenity_category_with_positive_size_or_value": amenity_flags,
            "amenity_flag_note": "Positive pool area or miscellaneous value can contradict an assumption that a missing category always means an absent amenity.",
        },
        "figures": [],
    }

    # 1: Compare dollar and natural-log distributions without filtering tails.
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.3), layout="constrained")
    for ax, series, color, xlabel, key in [
        (axes[0], price, BLUE, "Sale price (USD)", "sale_price_usd"),
        (axes[1], log_price, TEAL, "ln(SalePrice)", "ln_sale_price"),
    ]:
        ax.hist(series, bins=40, color=color, edgecolor="white", linewidth=0.55)
        ax.axvline(series.median(), color="#333333", linewidth=1.6, linestyle="--", label="Median")
        ax.set(xlabel=xlabel, ylabel="Number of homes")
        ax.set_title("Original target" if key == "sale_price_usd" else "Natural-log target", loc="left")
        ax.text(0.97, 0.95, f"n = {len(series):,}\nSample skewness = {series.skew():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.grid(axis="y", color="#e4e7eb", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="center right")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
    fig.suptitle("Sale prices have a long right tail; taking logs reduces asymmetry", fontsize=17, x=0.01, ha="left")
    save_figure(fig, figures_dir, "01_target_distributions")
    summary["figures"].append({"stem": "01_target_distributions", "population": "All 1460 training rows", "caption": "SalePrice in dollars and its natural logarithm, with 40 histogram bins and the sample median. Both panels retain all observations."})

    # 2: Top 20 raw NA/blank rates. Full list is preserved in the JSON summary.
    shown = list(reversed(missing[:20]))
    fig, ax = plt.subplots(figsize=(13.8, 8.1), layout="constrained")
    ypos = np.arange(len(shown))
    ax.barh(ypos-0.19, [r["train_percent"] for r in shown], height=0.36, color=BLUE, label="Train (n = 1,460)")
    ax.barh(ypos+0.19, [r["test_percent"] for r in shown], height=0.36, color=ORANGE, label="Test (n = 1,459)")
    ax.set_yticks(ypos, [r["feature"] for r in shown])
    ax.set(xlabel="Blank or NA entries (% of rows)", xlim=(0, 106))
    ax.set_title("Missingness concentrates in optional amenities", loc="left", fontsize=17, pad=15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e4e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    for i, row in enumerate(shown):
        if max(row["train_percent"], row["test_percent"]) >= 15:
            ax.text(max(row["train_percent"], row["test_percent"])+1, i,
                    f"{row['train_percent']:.1f} / {row['test_percent']:.1f}%", va="center", fontsize=9)
    fig.supxlabel("Top 20 features by the larger split rate. NA can mean no amenity; literal 'None' is preserved. Labels: train / test.", fontsize=10)
    save_figure(fig, figures_dir, "02_missingness")
    summary["figures"].append({"stem": "02_missingness", "population": "All train and test rows", "caption": "Top 20 features ranked by the larger train/test blank-or-NA percentage. The valid MasVnrType category None is not counted as missing. NA in amenity fields may represent structural absence."})

    # 3: Scatter/boxplots use all observations and distinguish observations from interpretations.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), layout="constrained")
    scatter = axes[0].scatter(train["GrLivArea"], price, c=train["OverallQual"], cmap="viridis",
                              vmin=1, vmax=10, s=25, alpha=0.62, linewidths=0)
    axes[0].set(xlabel="Above-ground living area (sq ft)", ylabel="Sale price (USD)")
    axes[0].set_title("Living area and sale price", loc="left")
    fig.colorbar(scatter, ax=axes[0], label="Overall quality (1–10)", shrink=0.82)
    for row in diagnostic.itertuples(index=False):
        axes[0].scatter(row.GrLivArea, row.SalePrice, s=110, facecolors="none", edgecolors="#A83327", linewidths=1.6)
        axes[0].annotate(f"Id {row.Id}", (row.GrLivArea, row.SalePrice), xytext=(-8, -22 if row.Id==1299 else 14),
                         textcoords="offset points", fontsize=9, ha="center", color="#8E291F")
    levels = sorted(train["OverallQual"].unique())
    grouped = [price[train["OverallQual"] == q].to_numpy() for q in levels]
    boxes = axes[1].boxplot(grouped, tick_labels=[str(q) for q in levels], patch_artist=True,
                            medianprops={"color":"#222222", "linewidth":1.4},
                            flierprops={"marker":"o", "markersize":3, "alpha":0.45},
                            widths=0.6)
    for box in boxes["boxes"]:
        box.set(facecolor="#9CC9C2", alpha=0.85)
    axes[1].set(xlabel="Overall quality rating", ylabel="Sale price (USD)")
    axes[1].set_title("Prices by overall quality", loc="left")
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
        ax.grid(axis="y", color="#e4e7eb", linewidth=0.65)
        ax.set_axisbelow(True)
    fig.suptitle("Area and overall quality relate to price, with substantial variation", fontsize=17, x=0.01, ha="left")
    fig.supxlabel("All 1,460 training homes. Red rings flag two large, low-price homes for inspection; neither row is removed. Boxplot whiskers: 1.5 × IQR.", fontsize=10)
    save_figure(fig, figures_dir, "03_area_quality_relationships")
    summary["figures"].append({"stem": "03_area_quality_relationships", "population": "All 1460 training rows", "caption": "Living area versus SalePrice, coloured by quality, and quality-group boxplots. Id 524 and 1299 satisfy GrLivArea > 4000 and SalePrice < 300000. Both are retained; these descriptive associations do not establish causality."})

    # 4: Domain-selected numeric examples, not a target-selected modelling feature set.
    heat_names = ["SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                  "GarageCars", "GarageArea", "YearBuilt", "YearRemodAdd", "FullBath", "TotRmsAbvGrd", "LotArea"]
    matrix = train[heat_names].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12.9, 10.1), layout="constrained")
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(heat_names)), heat_names, rotation=45, ha="right")
    ax.set_yticks(range(len(heat_names)), heat_names)
    for i in range(len(heat_names)):
        for j in range(len(heat_names)):
            value = matrix.iat[i,j]
            ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(value)>.64 else "#1B2530")
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.83)
    ax.set_title("Several size measures contain overlapping information", loc="left", fontsize=17, pad=18)
    ax.grid(False)
    fig.supxlabel("Predefined examples from size, quality and age variables. Original values, all 1,460 training rows; descriptive only, not a feature-selection rule.", fontsize=10)
    save_figure(fig, figures_dir, "04_numeric_correlations")
    summary["figures"].append({"stem": "04_numeric_correlations", "population": "All 1460 training rows; selected columns have no missing values", "caption": "Pearson correlations on original-valued, predefined numeric examples. Correlations describe these variables; preprocessing and feature selection follow the cross-validation protocol.", "columns": heat_names})

    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Verify saved numeric summaries against source and all expected figure outputs.
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["dataset"]["train_rows"] == len(train)
    assert saved["dataset"]["train_missing_cells"] == int(train.isna().sum().sum())
    assert saved["target"]["sale_price_usd"]["median"] == float(price.median())
    for figure in saved["figures"]:
        for suffix in ("png", "pdf"):
            assert (figures_dir / f"{figure['stem']}.{suffix}").stat().st_size > 1000
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "features")
    args = parser.parse_args()
    result = generate(args.data_dir, args.output_dir)
    print(json.dumps({
        "train_shape": [result["dataset"]["train_rows"], result["dataset"]["train_columns"]],
        "test_shape": [result["dataset"]["test_rows"], result["dataset"]["test_columns"]],
        "target_median_usd": result["target"]["sale_price_usd"]["median"],
        "target_skewness": result["target"]["sale_price_usd"]["skewness_sample"],
        "ln_target_skewness": result["target"]["ln_sale_price"]["skewness_sample"],
        "figures_created": len(result["figures"]),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
