"""Fit a small VAR grid on training data and rank common validation rollouts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import PROJECT_DIR, prepare_datasets, sha256_file
from evaluation import (ValidationEvaluator, conditioning_history, decode_batch, metric_specification,
                        validation_windows, validity_rates)
from modules import VARFit, lagged_pairs
from train import make_checkpoint
from ar_report import source_fingerprint


DEFAULT_GRID = {"lags": [1, 2, 4], "ridges": [1e-6, 1e-4, 1e-2], "seeds": [42, 43, 44]}
RANK_KEYS = ("invalid_rate_mean", "mean_kl_mean", "abs_return_acf_mae_mean", "lags", "ridge")


def select_candidate(rows):
    eligible = [row for row in rows if row["status"] == "ok"]
    if not eligible:
        raise RuntimeError("Every candidate failed; inspect per-seed errors before changing the grid.")
    return min(eligible, key=lambda row: tuple(row[key] for key in RANK_KEYS))


def experiment_config(lags, ridges, seeds, sequence_length, threads):
    return {"lags": list(lags), "ridges": list(ridges), "seeds": list(seeds),
            "sequence_length": sequence_length, "threads": threads}


def run_grid(data_dir, lags=(1, 2, 4), ridges=(1e-6, 1e-4, 1e-2), seeds=(42, 43, 44),
             sequence_length=32, threads=4, report_path=None, output_dir=None, render=True):
    if not lags or any(not isinstance(p, int) or p < 1 for p in lags):
        raise ValueError("Provide positive integer lag orders.")
    if not ridges or any(not np.isfinite(r) or r < 0 for r in ridges):
        raise ValueError("Provide finite non-negative ridge penalties.")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds):
        raise ValueError("Provide at least two distinct valid generation seeds.")
    if threads < 1:
        raise ValueError("threads must be positive.")
    if 1 not in lags or 1e-4 not in ridges:
        raise ValueError("Keep the original VAR(1), ridge=1e-4, in the comparison grid.")
    torch.set_num_threads(threads)
    report_path = Path(report_path or PROJECT_DIR / "reports/ar/results.json")
    output_dir = Path(output_dir or PROJECT_DIR / "outputs/ar/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    datasets, scaler, audit = prepare_datasets(data_dir, sequence_length, sequence_length)
    train, validation = datasets["train"], datasets["validation"]
    # Existing full-day loading/audit also prepares test data. No test outcomes are evaluated or used below.
    del datasets
    loading_seconds = perf_counter() - started
    starts, anchors, real = validation_windows(validation, max(lags))
    spec = metric_specification(train, max_lag=min(10, sequence_length - 1))
    train_values = torch.as_tensor(train.features, dtype=torch.float64)
    validation_values = torch.as_tensor(validation.features, dtype=torch.float64)
    reference = ValidationEvaluator(real, anchors, spec)
    rows, runs, curves, distributions = [], [], {}, {}
    selected_model, best_row = None, None
    # Small previews are always the first three eligible windows and first declared seed.
    previews = {"real": real[:3], "anchors": anchors[:3], "raw_start_indices": starts[:3] + validation.raw_offset}
    for order in lags:
        fitted_at = perf_counter()
        fitter = VARFit(train_values, order)
        setup_seconds = perf_counter() - fitted_at
        history = conditioning_history(validation, starts, order)
        x_val, y_val = lagged_pairs(validation_values, order)
        offset = max(lags) - order  # Identical one-step target rows for every order.
        for ridge in ridges:
            key = f"p{order}_ridge{ridge:g}"
            fitted_at = perf_counter()
            model = fitter.fit(ridge)
            solve_seconds = perf_counter() - fitted_at
            with torch.no_grad():
                validation_mse = float((model(x_val[offset:]) - y_val[offset:]).square().mean())
            row = {"candidate": key, "lags": order, "ridge": ridge, "status": "ok",
                   "fit_seconds": setup_seconds / len(ridges) + solve_seconds, "ridge_solve_seconds": solve_seconds,
                   "decomposition_seconds": setup_seconds, "training_one_step_mse": float(model.residuals.square().mean()),
                   "training_pairs": len(train_values) - order, "validation_one_step_mse": validation_mse,
                   "coefficient_count": model.weight.numel() + model.bias.numel(), "spectral_radius": model.spectral_radius}
            seed_rows, diagnostics = [], []
            for seed in seeds:
                run = {"candidate": key, "seed": seed, "status": "ok"}
                try:
                    generated_at = perf_counter()
                    features = model.generate(history, sequence_length, seed)
                    run["generation_seconds"] = perf_counter() - generated_at
                    books = decode_batch(features, scaler, anchors)
                    scores, details = reference.evaluate(books)
                    run.update(scores)
                    diagnostics.append(details)
                    if seed == seeds[0]:
                        previews[key] = books[:3]
                except (FloatingPointError, RuntimeError) as error:
                    run.update(status="failed", error=str(error))
                    row["status"] = "failed"
                seed_rows.append(run)
                runs.append(run)
            if row["status"] == "ok":
                metric_names = [name for name in seed_rows[0] if name not in ("candidate", "seed", "status")]
                for name in metric_names:
                    values = [run[name] for run in seed_rows]
                    row[f"{name}_mean"], row[f"{name}_sd"] = float(np.mean(values)), float(np.std(values, ddof=1))
                curves[key] = np.mean([d["acf"] for d in diagnostics], axis=0).tolist()
                distributions[key] = {name: np.mean([d[name] for d in diagnostics], axis=0).tolist()
                                      for name in reference.probabilities}
                if best_row is None or tuple(row[k] for k in RANK_KEYS) < tuple(best_row[k] for k in RANK_KEYS):
                    best_row = row
                    selected_model = model
            rows.append(row)
            print(f"{key}: {row['status']}; invalid={row.get('invalid_rate_mean', float('nan')):.2%}", flush=True)
        del fitter  # Do not retain the previous order's tall design during the next SVD.

    selected = select_candidate(rows)
    original = next(row for row in rows if row["lags"] == 1 and row["ridge"] == 1e-4)
    checkpoint = make_checkpoint(selected_model, train, scaler, {**selected, "fit_split": "train", "sequence_length": sequence_length})
    torch.save(checkpoint, output_dir / "selected_model.pt")
    np.savez_compressed(output_dir / "validation_examples.npz", **previews)
    report = {
        "format_version": 2, "config": experiment_config(lags, ridges, seeds, sequence_length, threads),
        "source_fingerprint": source_fingerprint(), "session": audit["session"], "splits": audit["splits"],
        "input_files": audit["files"], "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "fit_split": "train", "selection_split": "validation", "test_evaluated": False, "device": "cpu",
        "validation_windows": len(starts), "snapshots_per_seed": len(starts) * sequence_length,
        "raw_validation_anchor_indices": (starts + validation.raw_offset).tolist(),
        "validation_one_step_target_rows": len(validation.features) - max(lags), "metric_specification": spec,
        "selection_rule": "Lexicographic: lowest seed-mean invalid rate, then mean KL, then ACF MAE, then p and ridge.",
        "selected_candidate": selected["candidate"], "original_candidate": original["candidate"],
        "real_validity": validity_rates(real), "real_abs_return_acf": reference.acf.tolist(), "generated_abs_return_acf": curves,
        "real_histograms": {name: values.tolist() for name, values in reference.probabilities.items()},
        "generated_histograms": distributions, "decompositions": len(lags),
        "candidates": rows, "runs": runs, "data_loading_seconds": loading_seconds,
        "total_seconds": perf_counter() - started, "selected_checkpoint_sha256": sha256_file(output_dir / "selected_model.pt"),
        "limitations": [
            "Conditional 32-update simulations: observed validation history and price anchors are supplied at every window.",
            "All weights, residuals, scaling and histogram boundaries use training only; validation selects hyperparameters.",
            "Existing preprocessing reads/audits the full day; no test model metrics or test-based choices are computed.",
            "Invalid finite books remain in all distribution and ACF metrics. No clipping, sorting, rounding or repair.",
            "Seed standard deviations measure innovation sampling variability, not independent days or confidence intervals.",
            "KL depends on the declared binning and smoothing; 32-update ACF only measures short event-time dependence.",
            "Depth SSIM and final test comparison are pending. TimeGAN has not been trained or evaluated.",
            "The lowest invalid rate is ranked first even when another candidate has lower KL or one-step MSE.",
            "Candidate fit time includes an equal share of the order's SVD setup plus its own ridge solve/residual calculation.",
            "Rollout timing excludes decoding and IO; initial smoke timings are historical and reported separately.",
        ],
    }
    # Keep the earlier measured smoke test with the report, rather than refitting or relabelling its numbers.
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("initial_experiment", "historical_validation", "refactor_verification"):
            if key in previous:
                report[key] = previous[key]
    if render:
        from ar_report import example_diagnostics, save_figures
        report["example_diagnostics"] = example_diagnostics(report, previews)
        report["figures"] = save_figures(report, previews, report_path.parent)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lags", type=int, nargs="+", default=DEFAULT_GRID["lags"])
    parser.add_argument("--ridges", type=float, nargs="+", default=DEFAULT_GRID["ridges"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_GRID["seeds"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run_grid(args.data_dir, args.lags, args.ridges, args.seeds, threads=args.threads)
    print(f"Selected {report['selected_candidate']}; saved reports/ar/results.json (validation only).")


if __name__ == "__main__":
    main()
