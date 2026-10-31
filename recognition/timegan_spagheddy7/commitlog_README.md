# Development log

## 31 October 2026 - AR baseline

Added an autoregressive baseline in PyTorch to compare with TimeGAN later. The model takes previous market updates and generates 32 new updates, using supplied starting prices. The fitted model can be saved and loaded again to generate examples.

After the initial AR(1), the next step was to see whether tuning improved the results. The grid tested 1, 2 and 4 previous updates, with ridge values of 1e-6, 1e-4 and 1e-2. Each combination used three generation seeds (42, 43 and 44) and the same 1,080 validation windows. Fitting used the training period only.

VAR(4) with ridge 1e-6 was selected because it had the lowest rate of invalid orderbooks. The results below are averages across the three seeds.

| Measure | Original VAR(1) | Tuned VAR(4) |
| --- | ---: | ---: |
| Invalid snapshots | 92.924% | 91.380% |
| Spread KL | 0.1510 | 0.1323 |
| Mid-price return KL | 0.6058 | 0.5913 |
| Absolute-return ACF error, lags 1-10 | 0.01881 | 0.02597 |

The improvement was small. More than 91% of the tuned model's snapshots still broke at least one orderbook condition, mainly the ordering of prices across levels. Both KL scores improved slightly, but the autocorrelation error got worse. These results give a baseline for comparison, although the generated books are not reliable enough for market simulation. No test-set model results were used to choose the settings.

All the AR results were put into [ar_results.ipynb](ar_results.ipynb), including the initial checks, tuning grid, plots, three example periods and interpretation. The supporting results and figures are in [reports/ar/](reports/ar/). The notebook now loads the saved results instead of rerunning training each time. Model files and generated arrays stay out of Git.

The code was also cleaned up to avoid repeating the same matrix decomposition for each ridge value, decode sequences together and reuse the real validation statistics. Over three runs per version, median grid time fell from 11.56 to 7.79 seconds, about 33% faster. This includes loading and saving, but excludes plotting and Python startup. The selected model and validity/KL scores stayed the same; changes in ACF error were below 1.4e-14.

Checks completed:

- All 20 tests passed, covering the model, data handling and evaluation functions.
- All 12 notebook code cells ran without errors. The plots and equation formatting were checked.
- The notebook ran from the saved report files without the dataset or model weights.
- The original saved AR(1) model reproduced the same generated features and decoded books with the recorded seeds and four CPU threads.

Depth SSIM and GPU training memory have not been measured yet. The full-day data audit reads the test period, but final model testing is still pending. Seed variation here does not show how the model would perform on other trading days.

Next steps:

1. Finish the SSIM measure and decide what starting information each model gets for a fair comparison. Confirm the tutor feasibility check-off.
2. Implement TimeGAN and run a small GPU trial to check training, memory use and saved-model generation.
3. Train with several seeds and compare against AR. Test the effect of removing the temporal supervision loss.
4. Once the settings are fixed, evaluate the test period and discuss 3-5 example failures.
5. Finish the README, feasibility section, references and disclosure, then submit the PDF and course pull request. Respond to the teaching team's review when it arrives.

## 26 October 2026 - data checks

Added [data_checks.ipynb](data_checks.ipynb) to record the preliminary checks for the feasibility review. The two CSVs contained 269,748 matching rows, with 6 columns in the message file and 40 in the orderbook file. The file hashes matched the saved audit, and the checks found no missing levels, non-positive spreads, incorrectly ordered prices or negative quantities.

The data was split 70/15/15 by elapsed trading time, with scaling fitted on training data only. Using 32 updates per sequence gave 5,507 training, 1,081 validation and 1,839 test sequences. A training-data plot and the notebook requirements were saved. The writing template was moved to feasibility_README.md and SETUP_NOTES.md was removed.

All 9 notebook code cells and 5 dataset tests passed. A batch of shape (64, 32, 40) transferred to the RTX 4060 successfully. This checked data loading and GPU compatibility; it did not measure training memory. The batch itself used 0.3125 MiB.

At this stage, neither model had been trained and the feasibility review was still a template awaiting tutor check-off. The next step was to fit and generate with the AR baseline.

## Git commits

| Commit | Work recorded |
| --- | --- |
| 6fbd4c9 | Data pipeline and initial feasibility checks. |
| 4f9a67f | AR baseline, validation tuning and results notebook. |

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
