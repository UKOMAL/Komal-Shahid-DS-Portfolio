# Project 2 — AI-Driven Content Strategy Engine (DSC680)

Reproducible code for the M2/M3 analysis.

## Modules

| File | Role |
|------|------|
| `shahid_dsc680_project2_data_collection.py` | Real Reddit (public JSON endpoint) + `trendspyg` + BeautifulSoup blog scraper. Import-safe; no I/O until called. |
| `shahid_dsc680_project2_synthetic_corpus.py` | Calibrated synthetic Reddit-like corpus used for the M2 analysis (live API pulls are rate-limited and not reproducible across grading runs). |
| `shahid_dsc680_project2_features.py` | Feature engineering: one-hot encoding, log transforms, design matrix. |
| `shahid_dsc680_project2_analysis.py` | End-to-end analysis: regression, ANOVA, lead-lag, generation evaluation, figures, results JSON. |

## Run

```bash
python -m pip install -r requirements.txt
python shahid_dsc680_project2_analysis.py
```

Outputs:

- `../figures/fig0[1-5]_*.png` — the five analysis figures embedded in the whitepaper.
- `../results/m2_results.json` — every reported metric, regenerated from scratch each run.
- `../results/corpus_sample.csv` — the analyzed corpus.

## Coding practices

- All categorical/numeric splits are constants at module top so they can be reused in generation.
- Models are wrapped in a `ModelReport` dataclass so reporting code stays decoupled from training.
- All random work uses a seeded `numpy.random.default_rng` — same seed, same numbers.
- I/O is centralized in `data_collection.py`; `analysis.py` operates on DataFrames only.
- `matplotlib.use("Agg")` so the script runs on headless CI / graders' machines.
- XGBoost falls back to scikit-learn's `GradientBoostingRegressor` if XGBoost isn't installed, keeping the pipeline runnable without GPU-class tooling.
