# TimeGAN for AMZN Limit Order Book Simulation

COMP3710 Section 2.7. The autoregressive baseline is implemented and evaluated on validation data. TimeGAN and final test evaluation are pending.

Start with [ar_results.ipynb](ar_results.ipynb) for the complete AR baseline report: initial fit, tuning, structural validity, distributions, temporal behaviour, depth SSIM, fixed examples, runtime and conclusions. All displayed results are saved; opening or rerunning the notebook does not train models by default.

Under the main `anchor_only_v1` protocol, VAR(2), ridge 0.01, reduces invalid validation snapshots from 93.15% to 93.02%. Depth SSIM rises from 0.125 to 0.151, below the > 0.6 target. Return KL improves, while spread KL and temporal error worsen. This is an inadequate order-book simulator; the small validity difference does not establish a reliable advantage.

The earlier experiment used observed validation histories. Its selected VAR(4), ridge 1e-6, still scores 91.38% invalid snapshots and now has measured SSIM 0.262. These supplementary results remain in the same notebook under `observed_history_v1`; they are not the main comparison against noise-driven TimeGAN.

## Shared comparison protocol

Both models learn only from training data. At generation time:

| Information | AR | TimeGAN (planned) |
| --- | --- | --- |
| Initial feature state | Random training history; last p rows of a shared four-row sample | Noise |
| Validation feature history or volumes | Not supplied | Not supplied |
| Starting prices | 20 observed prices, supplied only to the common decoder | Same prices and decoder |
| Evaluation | 1,080 chronological validation windows, 32 updates, seeds 42/43/44 | Same windows, length, seeds and metrics |

AR history sampling uses seed + 10000, independently of residual sampling. The two models need not use identical random draws; the seed schedule and evaluation conditions must match. Generated sample i is paired with real window i, without best-of-many matching or structural repair. This evaluates anchored feature generation, not forecasting a specific observed future or generating starting prices. `protocol.py` and the saved comparison signature record the contract for the later TimeGAN evaluation.

Depth SSIM compares separate 10-level × 32-update ask/bid volume images. Signed log volumes are divided by the maximum log volume in training; `data_range=1` remains fixed and no values are clipped. We use 7×7 uniform windows, sample covariance, K1=0.01 and K2=0.03, averaging valid local windows, sides, sequences and seeds. This is a declared variant of the [SSIM equation](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), not the original 11×11 Gaussian setting. It measures volume similarity, not valid prices; the fixed shuffled-real control illustrates pairing sensitivity. SSIM is reported but does not affect grid selection.

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

The first command fits nine configurations once and evaluates each with three seeds under both protocols: 54 evaluations, one results bundle. Then rerun the notebook to refresh its displayed outputs. Alternatively, set `RECOMPUTE_GRID = True` in the notebook. The historical initial smoke-test measurements remain labelled separately. The default notebook view needs neither raw CSVs nor checkpoints.

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
| `evaluation.py` | Strict book validity, KL, within-window ACF and depth SSIM; cached real reference statistics |
| `protocol.py` | Training-only initial histories and the shared AR/TimeGAN information contract |
| `tune_ar.py` | Shared fits, both initial-information protocols and explicit validation selection |
| `ar_report.py` | Report loading, provenance checks and plots |
| `reports/ar/` | One report bundle: results.json and report figures, including historical measurements |
| `outputs/ar/` | Ignored checkpoints and generated arrays; selected model under `tuning/` |

The CSVs remain outside Git. No book repairs are applied. The full-day data audit reads test data, but test model scores are not computed or used for tuning. The TimeGAN comparison and final held-out evaluation remain pending.

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
