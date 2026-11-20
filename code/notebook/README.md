# Notebook workspace

`prepare.py` copies the Notebook and its required files into `outputs/notebook/`, preserving the repository's `code/`, `config/`, `data/` and `notebooks/` layout. [config/notebook.json](../../config/notebook.json) lists the current source paths and SHA256 checksums.

From the repository root, prepare an interactive copy:

```sh
python3 code/notebook/prepare.py --prepare
```

Open `outputs/notebook/notebooks/house_prices.ipynb` in the environment installed from [requirements.txt](../../requirements.txt). To execute the copy from the command line:

```sh
python3 code/notebook/prepare.py --execute
```

The Notebook uses [model.py](../ensemble/model.py) to recompute the ensemble from saved component predictions. Its data and prediction checks use the original experiment outputs.

For another execution, choose a new workspace with `--output outputs/notebook_rerun`. Existing execution results and modified workspace files are protected from overwrite.
