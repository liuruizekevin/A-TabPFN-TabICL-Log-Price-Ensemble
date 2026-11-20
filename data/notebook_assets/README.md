# Notebook resources

`assets.json` stores the final submission screenshot used by the Notebook. The preparation command extracts its preserved bytes into the configured `data/` path inside the Notebook workspace. EDA figures are extracted from the Notebook's saved image outputs, with their checksums verified against [config/notebook.json](../../config/notebook.json).

The prepared Notebook is `outputs/notebook/notebooks/house_prices.ipynb`. See the [workspace instructions](../../code/notebook/README.md) for preparation and execution commands.
