# Development log

## 31 October 2026 - autoregressive baseline and validation report

Completed:

- Implemented a PyTorch vector AR(p) baseline with ridge fitting, joint residual sampling, checkpoint saving/reloading and generation of 32-update sequences.
- Compared p = 1, 2, 4 and ridge = 1e-6, 1e-4, 1e-2 using training-only fits and generation seeds 42, 43 and 44.
- Evaluated 1,080 common validation windows per seed (34,560 snapshots), using observed past validation features and matching price anchors. No test model scores were used.
- Selected VAR(4), ridge 1e-6, by lowest mean invalid-book rate, with mean KL and ACF error as tie-breakers. Kept the original VAR(1) comparison.
- Consolidated the initial smoke test, grid, per-seed scores, structural diagnostics, distributions, temporal behaviour, three fixed example periods, depth heatmaps and runtime evidence into [ar_results.ipynb](ar_results.ipynb).
- Saved current and historical numeric evidence plus six figures in [reports/ar/](reports/ar/). The notebook loads saved results by default; weights and generated arrays remain ignored under outputs/ar/.
- Shared one SVD per lag order across ridge values, vectorised sequence decoding, cached real validation statistics and reused the checkpoint schema. Removed the superseded report copies after checking their evidence was preserved.
- Corrected notebook equation delimiters and separated the model equation from the fitting objective.

Validation results (means over three generation seeds; lower is better):

| Measure | Original VAR(1), ridge 1e-4 | Selected VAR(4), ridge 1e-6 |
| --- | ---: | ---: |
| Invalid snapshots | 92.924% | 91.380% |
| Spread KL | 0.1510 | 0.1323 |
| Mid-price return KL | 0.6058 | 0.5913 |
| Absolute-return ACF error, lags 1-10 | 0.01881 | 0.02597 |

Interpretation: tuning modestly improves validity and distribution similarity, but worsens the temporal score. More than 91% of generated snapshots violate at least one constraint across the ten levels. The AR model remains a benchmark rather than an adequate simulator. KL values depend on the declared training-only bins and smoothing; seed standard deviations measure generation randomness, not confidence across trading days.

Verification:

- All 20 regression checks passed, including synthetic VAR recovery, lag ordering, checkpoint/RNG behaviour, chronological data handling, batch decoding, histogram tails and strict order-book validity.
- All 12 report notebook code cells executed without errors. Displayed figures were inspected; Markdown changes preserved code and outputs.
- The notebook also executed from a report-only copy without raw CSVs, checkpoints or ignored preview arrays, without modifying the saved report bundle.
- The original saved AR(1) checkpoint reproduced its features and decoded books exactly using the recorded seeds and four CPU threads.
- Before/after comparison of all 27 validation evaluations retained the same selected configuration and identical validity/KL scores. The largest ACF-score difference was approximately 1.32e-14.
- Across three runs per implementation on the same CPU, median grid runtime fell from 11.563 s to 7.794 s (32.6% faster); fitting work fell from 5.077 s to 2.435 s. Timings include data loading/saving but exclude plotting and interpreter startup.

Limits: validation was used for model selection; final test evaluation remains pending. The full-day preprocessing audit reads test data, but no test model scores were computed. Depth SSIM, TimeGAN training, a fair cross-model conditioning protocol and GPU training-resource measurements remain to be completed.

Next milestones:

1. Finalise the shared comparison protocol and depth-SSIM definition; confirm the feasibility tutor check-off.
2. Implement TimeGAN's five networks and staged training, with a small GPU pilot, checkpoint reload checks and resource measurements.
3. Run repeated-seed training and validation comparisons; use a targeted temporal-loss ablation to investigate the research question.
4. Freeze model choices, evaluate the held-out test period and analyse 3-5 representative failures.
5. Complete the final README, feasibility section, references/disclosure, reproducibility checks and PDF; submit the course pull request and Turnitin report, then respond to the later code review.

## 26 October 2026 - data checks and feasibility preparation

Completed:
- Added an executed notebook explaining and reproducing the preliminary data checks.
- Checked all 269,748 aligned rows in the AMZN message and Level-10 order-book files. The message file has 6 columns; the order-book file has 40.
- Confirmed the file hashes match the saved audit and found no violations in the implemented data-integrity checks.
- Documented chronological splits by elapsed session time (70/15/15), training-only scaling, and sequences of 32 market updates.
- Confirmed 5,507 training, 1,081 validation and 1,839 test sequences.
- Saved a training-data plot and the notebook dependency list.
- Separated the report-writing template into feasibility_README.md and removed the earlier SETUP_NOTES.md.

Verification recorded in data_checks.ipynb:
- All 9 code cells executed without errors.
- All 5 dataset regression tests passed, covering split boundaries, scaling, inverse transformation and invalid input rejection.
- A finite batch of shape (64, 32, 40) transferred to the RTX 4060 successfully.

Limits: this verifies the data pipeline and a GPU batch transfer. Neither TimeGAN nor the autoregressive baseline has been trained. The input batch uses 0.3125 MiB; this is not peak training memory. The feasibility review remains a writing template and has not been checked off.

Next: Implement the autoregressive baseline, then run a small TimeGAN training pilot and measure training memory and runtime.

## Existing Git milestones

| Commit | Recorded change |
| --- | --- |
| 6fbd4c9 | Initial project snapshot with data pipeline and feasibility checks; lecturer course history retained. |

The 31 October entry accompanies the commit titled `Add autoregressive baseline and validation report`. Subsequent milestones will be recorded as they are completed.

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
