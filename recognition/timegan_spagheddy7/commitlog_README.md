# Development log

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

Next: complete the feasibility review, implement the autoregressive baseline, then run a small TimeGAN training pilot and measure training memory and runtime.

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
