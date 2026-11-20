# Training and prediction code

Training and prediction scripts for the six experiment rounds. Shared algorithms are in shared/, and the selected model is in ensemble/.

Experiment commands run in disposable workspaces with `features/`, `models/` and `submissions/` directories. `data/catalog.json` maps these normalized experiment paths to retained repository files, and `code/registry.json` maps the executable scripts. Generated results stay under `outputs/iteration_XX/`.

| Folder | Contents |
| --- | --- |
| [ensemble](ensemble/) | Selected ensemble prediction code |
| [iteration_01](iteration_01/) | Baseline training and feature engineering |
| [iteration_02](iteration_02/) | Model tuning and blend selection |
| [iteration_03](iteration_03/) | Follow-up models and blend selection |
| [iteration_04](iteration_04/) | Classical models, TabPFN and stacking |
| [iteration_05](iteration_05/) | TabPFN variants and ensemble code |
| [iteration_06](iteration_06/) | TabPFN and TabICL experiment code |
| [notebook](notebook/) | Original Notebook workspace |
| [shared](shared/) | Shared experiment routines |
