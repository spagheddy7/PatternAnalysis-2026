# TimeGAN for AMZN Limit Order Book Simulation

COMP3710 Section 2.7 project in progress. The data pipeline and preliminary checks are available; model implementation and training are still pending.

- [Feasibility review and report draft](feasibility_README.md): the writing template.
- [Data checks notebook](data_checks.ipynb): executed checks, explanations and plots.
- [Development log](commitlog_README.md): completed work, verification and next steps.
- [Saved data audit](reports/data_audit.json): measured preprocessing results.

The data contains one AMZN trading session, 21 June 2012. Chronological preprocessing produces 5,507 training, 1,081 validation and 1,839 test sequences, each containing 32 updates and 40 features. These are data checks, not evidence of model performance.

## Run the data checks

From the repository root, with the project Python environment active:

```powershell
python recognition/timegan_spagheddy7/dataset.py --data-dir "../LOBSTER_SampleFile_AMZN_2012-06-21_10" --require-cuda
python -m unittest discover -s recognition/timegan_spagheddy7 -p "test_*.py" -v
```

Use the same environment as the notebook kernel. Core dependency versions are listed in requirements.txt; notebook extras are in requirements-notebook.txt. The downloaded CSV files remain outside the repository.

## Artificial Intelligence Usage Disclosure

I used AI models to assist in writing the code for the data_checks notebook as well as functions in the dataset.py script and test_dataset.py
