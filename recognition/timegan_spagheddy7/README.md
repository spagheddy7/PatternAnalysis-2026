# TimeGAN for AMZN Limit Order Book Simulation

COMP3710, Section 2.7, Hard difficulty. GitHub account: `spagheddy7`.

## Project question and current status

Can TimeGAN generate AMZN Level-10 order-book sequences that reproduce spread
and return distributions while maintaining valid price ladders and volumes,
compared with a simple autoregressive generator?

This first milestone implements data loading, chronological splitting,
invertible preprocessing, lazy PyTorch sequence windows, a data audit, and a
GPU minibatch check. **TimeGAN and the baseline are not implemented or trained
yet.** The feasibility review and compulsory tutor check-off are outstanding.

## Repository and submission

- Fork: <https://github.com/spagheddy7/PatternAnalysis-2026>
- Course repository: <https://github.com/shakes76/PatternAnalysis-2026>
- Work and commit on `topic-recognition`; push to the personal fork (`origin`).
- The final pull request targets the course repository's `topic-recognition`.
- Keep project code and report in `recognition/timegan_spagheddy7/`.
- Never commit raw data, processed training arrays, model weights, or environments.
- Final submission also requires a PDF export of this report through Turnitin.

## Environment and commands

Tested core versions are pinned in `requirements.txt`: Python 3.12, PyTorch
2.8.0 (local CUDA 12.8 wheel), NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.7.2,
and Matplotlib 3.10.7. The local `.venv` sits in the parent assignment directory,
outside the Git checkout. It was created using Codex's bundled Python 3.12;
reproduction can use a normal Python 3.12 installation without Codex.

For a new environment, from the repository root in PowerShell:

```powershell
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
..\.venv\Scripts\python.exe -m pip install -r recognition\timegan_spagheddy7\requirements.txt
```

Select the appropriate PyTorch wheel for another computer using the official
installation instructions. PyG is not needed by this project.

The local AMZN sample is stored separately at
`../../Project/2.7/LOBSTER_SampleFile_AMZN_2012-06-21_10/` relative to this
repository. Pass the directory containing the two extracted CSVs:

```powershell
..\.venv\Scripts\python.exe recognition\timegan_spagheddy7\dataset.py --data-dir "..\..\Project\2.7\LOBSTER_SampleFile_AMZN_2012-06-21_10" --require-cuda
..\.venv\Scripts\python.exe -m unittest discover -s recognition\timegan_spagheddy7 -p "test_*.py" -v
```

The audit writes `reports/data_audit.json` and refreshes the training-only
preview below. Dataset paths are supplied at runtime, not hard-coded into the
loader. Raw files remain in their original directory.

## Data representation and leakage prevention

The LOBSTER sample contains one trading day, AMZN on 21 June 2012. The message
and order-book files are aligned by row. Order-book columns repeat
`ask price, ask shares, bid price, bid shares` across ten levels. Raw prices are
USD multiplied by 10,000. The loader checks dimensions, chronological order,
missing-level sentinel values, non-negative volumes, positive spreads, price
ladder ordering, and trading halts. Unsupported invalid rows cause an explicit
error rather than being silently dropped or concatenated across gaps.

**Split before preprocessing:** use the first 70% of session clock time for
training, the next 15% for validation, and the final 15% for testing. The
cutoffs are 14:03:00 and 15:01:30 in the dataset's market clock. These are
dataset timestamps, not Brisbane times. Equal-time events stay in one split.
This is a time split, so the fractions of rows need not equal 70/15/15.

Within each split, transform the 20 price series into event-to-event log
returns, and the 20 volume series using `log1p`. Fit per-feature min-max
scaling on training transitions only. Do not clip future values outside the
training range. The first state of each split is an anchor, so no return or
window crosses a split boundary.

The initial configuration uses 32-event sequences with stride 32 and 40
features. Each sequence stores its 20 starting prices to allow reconstruction
into dollars and shares. Adjacent windows within a split can share their
boundary anchor state, but their target transitions do not overlap. Windows
are sliced lazily to avoid materialising a large duplicated array.

`decode_features` reverses the scaling and transformations without clipping
volumes or sorting prices. This preserves possible generated violations for
honest later evaluation. Starting prices are conditioning information; this
does not claim unconditional generation of a whole trading day.

![Initial training data preview](figures/training_data_preview.png)

## Preliminary evidence and limitations

The first audit of the supplied sample found 269,748 aligned rows, no missing
price levels, no negative volumes, no nonpositive spreads or disordered price
ladders, and no trading-halt rows. With the initial 32-event/stride-32 setup:

| Split | Raw book states | 32-event sequences |
| --- | ---: | ---: |
| Training | 176,255 | 5,507 |
| Validation | 34,624 | 1,081 |
| Test | 58,869 | 1,839 |

The RTX 4060 Laptop GPU successfully accepted a `64 x 32 x 40` training batch.
All five regression checks passed. The first training sequence reconstructed
with maximum absolute errors of approximately `2.59e-7` USD and `0.00128`
shares, reflecting float32 feature storage rather than a model error.

See `reports/data_audit.json` for source-file SHA-256 hashes, raw integrity
checks, split sizes, sequence counts, training-scaler parameters, numerical
reconstruction errors, and the actual device check. The regression checks
cover split/window separation, equal timestamps, training-only scaling,
reconstruction, mismatched files, and missing price levels.

The GPU check transfers one training minibatch and checks finite values. Its
input size is **not** a TimeGAN training-memory or runtime estimate. No final
test-set model evaluation has occurred.

This is one stock on one day. A later intraday test split cannot demonstrate
cross-day or cross-stock generalisation. Sequences are in event time, so the
future generator does not yet model waiting times between market events.

## Next milestones

1. Prepare the one-page feasibility review and arrange the tutor check-off.
2. Implement the baseline and TimeGAN components in `modules.py` and training
   in `train.py`, including a small measured training experiment.
3. Choose settings using validation data, then evaluate on the held-out test
   period. Report spread/return KL divergence, depth-heatmap SSIM, price and
   volume constraint violations, volatility clustering, computational costs,
   and three to five representative real/generated comparisons.
4. Add `predict.py`, results, the engineering recommendation, and reproducible
   inference instructions. Keep weights outside Git.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with interpreting the specification, project planning,
GitHub and environment setup, and drafting this data pipeline, regression
checks, documentation, and preview. Verification consists of executing the
pipeline on the supplied CSVs, checking the supplied LOBSTER format notes,
testing chronology and training-only preprocessing on synthetic cases, and
testing reconstruction and GPU batch transfer. The student must review and
understand the implementation. Update this disclosure as further work occurs.

## References

- The course's draft COMP3710/COMP7710 report specification, Section 2.7.
- `LOBSTER_SampleFiles_ReadMe.txt`, supplied with the AMZN sample (1 September 2013).
- [LOBSTER data format](https://lobsterdata.com/info/DataStructure.php)
- [LOBSTER sample data](https://lobsterdata.com/info/DataSamples.php)
- [TimeGAN authors' reference repository](https://github.com/jsyoon0823/TimeGAN)
- [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
