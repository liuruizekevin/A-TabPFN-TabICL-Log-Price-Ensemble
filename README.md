# House Price Prediction

**From classical regression to a TabPFN–TabICL ensemble**

[Report](documents/report.pdf) · [Slides](documents/slides.pdf) · [Notebook](notebooks/house_prices.ipynb) · [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

DSAI4204 Data Mining & Data Warehousing

## Overview

This project studies house-price prediction on the Ames Housing dataset through feature engineering, regression models and prediction ensembles. We compare classical machine-learning methods with pretrained tabular foundation models, then combine complementary predictions in log-price space.

The competition data contain **1,460 training samples**, **1,459 test samples** and **79 predictors** describing property characteristics. Across **six experimental rounds**, we evaluate **350 configurations** and record **36 Kaggle submissions**. The selected TabPFN–TabICL ensemble achieves a Kaggle score of **0.10970**.

The repository includes the experiment code, data, saved predictions, submission records and an executable Notebook accompanying the project report.

## Method

Our experiments examine three parts of the prediction pipeline:

1. **Data preparation:** target transformations, missing-value handling, categorical features and domain-derived housing features.
2. **Model comparison:** linear regression models, support vector regression, gradient boosting, stacking, TabPFN and TabICL.
3. **Ensembling and evaluation:** blending predictions and comparing five-fold log RMSE across candidate models.

The selected model combines two feature views of TabPFN with TabICLv2:

| Component | Input representation | Weight |
| --- | --- | ---: |
| Raw TabPFN | Original predictors | 0.64 |
| Domain-feature TabPFN | Predictors with domain-derived features | 0.16 |
| TabICLv2 | Original predictors | 0.20 |

The final prediction is a weighted average in natural-log space, converted back to the original price scale:

$$
\widehat{y}=\exp\left(0.64\,z_{\mathrm{raw}}+0.16\,z_{\mathrm{domain}}+0.20\,z_{\mathrm{TabICL}}\right),
$$

where each $z$ is a component's predicted log price. The implementation and weights are available in [the ensemble code](code/ensemble/) and [model configuration](config/model.json).

## Results

| Metric | Selected ensemble |
| --- | ---: |
| Kaggle score | **0.10970** |
| Mean five-fold log RMSE | 0.115308 |
| Sample standard deviation across folds | 0.021422 |
| Pooled out-of-fold log RMSE | 0.116889 |

The development folds and leaderboard feedback were reused during model selection. The Kaggle score belongs to the preserved submitted CSV.

## Experiments

| Round | Configurations | Focus |
| --- | ---: | --- |
| [01](code/iteration_01/) | 21 | Target scale, missing values, feature engineering and baseline estimators |
| [02](code/iteration_02/) | 29 | Classical-model tuning and prediction blends |
| [03](code/iteration_03/) | 8 | Follow-up blends and row-filter comparisons |
| [04](code/iteration_04/) | 263 | Extended classical models, TabPFN and stacking |
| [05](code/iteration_05/) | 17 | TabPFN preprocessing and two-view ensembles |
| [06](code/iteration_06/) | 12 | TabPFN–TabICL blending and candidate comparisons |

The [report](documents/report.pdf) presents the experimental analysis. The [Notebook](notebooks/house_prices.ipynb) explains the algorithms, data preparation and saved-prediction workflow.

## Getting Started

Clone the repository, then use Python 3.12 to verify the selected result and recompute the ensemble from saved component predictions:

```sh
git clone https://github.com/liuruizekevin/A-TabPFN-TabICL-Log-Price-Ensemble.git
cd A-TabPFN-TabICL-Log-Price-Ensemble
python3.12 code/ensemble/model.py verify
python3.12 code/ensemble/model.py predict
```

These commands use the Python standard library to combine the saved component predictions. Recomputed predictions are written to `outputs/predictions.csv`. To export the exact CSV associated with the recorded Kaggle score, run `python3.12 code/ensemble/model.py export`.

For the interactive workflow, install [requirements.txt](requirements.txt) and prepare a Notebook workspace:

```sh
python3 code/notebook/prepare.py --prepare
```

Open `outputs/notebook/notebooks/house_prices.ipynb`. Its workspace uses the same directory layout as the repository; the [Notebook instructions](code/notebook/README.md) also explain command-line execution. The default workflow combines saved component predictions.

Dependencies for the six experiment rounds are listed in [requirements-experiments.txt](requirements-experiments.txt). Pretrained checkpoints are supplied separately.

## Repository Structure

```text
code/          Training, prediction and evaluation code
config/        Model settings and checkpoint specifications
data/          Competition data, experiment results and saved predictions
documents/     Project report and presentation slides
notebooks/     Algorithm explanations and executable data workflow
tests/         Data integrity and prediction checks
```

## Acknowledgements

This project builds on [TabPFN](https://github.com/PriorLabs/TabPFN) and [TabICL](https://github.com/soda-inria/tabicl). We thank their authors for releasing the models and implementations, and Kaggle for hosting the House Prices competition.
