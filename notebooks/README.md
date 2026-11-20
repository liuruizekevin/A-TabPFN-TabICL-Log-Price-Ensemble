# Algorithms and data workflow

[house_prices.ipynb](house_prices.ipynb) opens with the project title, team name, overview and running instructions, followed by a plain-text contents list and two parts:

- Part A: Algorithms and the prediction procedure.
- Part B: Eight numbered steps covering setup, data exploration, feature preparation, prediction and validation.

The contents list uses plain text. The Notebook has 25 cells, including seven code cells with saved outputs. Full experiment results are kept in [data](../data/).

[config/notebook.json](../config/notebook.json) lists the Notebook file, required repository paths and SHA256 checksums. Preparation copies those files into a workspace with the same directory layout as the repository:

```sh
python3 code/notebook/prepare.py --prepare
```

Open `outputs/notebook/notebooks/house_prices.ipynb`. To execute the copy from the command line:

```sh
python3 code/notebook/prepare.py --execute
```

The Notebook recomputes the ensemble from saved component predictions with [model.py](../code/ensemble/model.py) and checks the original data, features and selected prediction outputs.
