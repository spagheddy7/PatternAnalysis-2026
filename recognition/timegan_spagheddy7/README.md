# TimeGAN for AMZN Limit Order Book Simulation

COMP3710 Section 2.7. The autoregressive baseline is implemented and evaluated on validation data. TimeGAN and final test evaluation are pending.

Start with [ar_results.ipynb](ar_results.ipynb) for the complete AR baseline report: initial fit, tuning, structural validity, distributions, temporal behaviour, fixed examples, runtime and conclusions. All displayed results are saved; opening or rerunning the notebook does not train models by default.

The selected VAR(4), ridge 1e-6, reduces invalid validation snapshots from 92.92% to 91.38%. Both KL scores improve slightly, but temporal error worsens. This is a useful benchmark, not an adequate order-book simulator. The report distinguishes these validation results from the earlier training-context smoke test.

## Project documents

- [AR baseline report](ar_results.ipynb): the main results notebook.
- [Data checks](data_checks.ipynb): data feasibility evidence.
- [Feasibility and report draft](feasibility_README.md): the student's writing.
- [Development log](commitlog_README.md): recorded work and verification.

## Reproduce the results

Use the project Python environment. Dependencies are in requirements.txt; notebook extras are in requirements-notebook.txt. From the repository root:

```powershell
python recognition/timegan_spagheddy7/tune_ar.py --data-dir "../LOBSTER_SampleFile_AMZN_2012-06-21_10"
python -m unittest discover -s recognition/timegan_spagheddy7 -p "test_*.py" -v
```

The first command recomputes the nine-configuration, three-seed validation grid and its figures. Then rerun the notebook to refresh its displayed outputs. Alternatively, set `RECOMPUTE_GRID = True` in the notebook. The historical initial smoke-test measurements remain labelled separately. The default notebook view needs neither raw CSVs nor checkpoints.

For standalone fitting and generation:

```powershell
python recognition/timegan_spagheddy7/train.py --data-dir "../LOBSTER_SampleFile_AMZN_2012-06-21_10" --lags 1
python recognition/timegan_spagheddy7/predict.py --num-sequences 64 --seed 42
```

These write local scratch results under `outputs/ar/`. They do not overwrite the published report. Generation samples training histories and produces 32 new updates after each supplied anchor.

## File responsibilities

| File or folder | Purpose |
| --- | --- |
| `dataset.py` | Full-day checks, chronological splits, training-only scaling and batch decoding |
| `modules.py` | PyTorch VAR model; shares one SVD across ridge values of the same order |
| `train.py` / `predict.py` | Standalone fit and reload/generate commands; shared checkpoint schema |
| `evaluation.py` | Strict book validity, KL and within-window ACF; cached real reference statistics |
| `tune_ar.py` | Training-only fitting, common validation contexts and explicit model selection |
| `ar_report.py` | Report loading, provenance checks and plots |
| `reports/ar/` | One report bundle: results.json and report figures, including historical measurements |
| `outputs/ar/` | Ignored checkpoints and generated arrays; selected model under `tuning/` |

The CSVs remain outside Git. No book repairs are applied. The full-day data audit reads test data, but test model scores are not computed or used for tuning. Depth SSIM, the TimeGAN comparison and final held-out evaluation remain pending.

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
