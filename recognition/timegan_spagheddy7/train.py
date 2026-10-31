"""Fit the autoregressive baseline using only the chronological training period."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from dataset import FEATURE_NAMES, PRICE_COLUMNS, PRICE_DIVISOR, PROJECT_DIR, prepare_datasets
from modules import VectorAR
from evaluation import conditioning_history


def make_checkpoint(model, training, scaler, report):
    """One checkpoint schema for standalone fits and validation-selected fits."""
    starts = training.starts[training.starts >= model.lags]
    if len(starts) == 0:
        raise ValueError("No training windows have enough conditioning history.")
    contexts = {"features": conditioning_history(training, starts, model.lags),
                "initial_prices": torch.as_tensor(training.book[starts][:, PRICE_COLUMNS] / PRICE_DIVISOR),
                "raw_start_indices": torch.as_tensor(starts + training.raw_offset, dtype=torch.int64)}
    return {"format_version": 2, "model_type": "vector_ar", "state_dict": model.state_dict(),
            "scaler": {"minimum": torch.from_numpy(scaler.minimum), "maximum": torch.from_numpy(scaler.maximum)},
            "feature_names": FEATURE_NAMES, "contexts": contexts, "fit_report": report}


def fit_baseline(data_dir, checkpoint_path, report_path, sequence_length=32, ridge=1e-4, threads=4, lags=1):
    if threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    started = perf_counter()
    datasets, scaler, audit = prepare_datasets(data_dir, sequence_length=sequence_length, stride=sequence_length)
    training = datasets["train"]
    features = torch.from_numpy(training.features)
    data_seconds = perf_counter() - started

    # Need p observed features inside this split, ending at features[start - 1].
    # That final feature describes book[start], whose prices anchor the rollout.
    starts = training.starts[training.starts >= lags]
    if len(starts) == 0:
        raise ValueError("Need at least two training windows to supply valid generation contexts.")
    started = perf_counter()
    model = VectorAR.fit(features, ridge=ridge, lags=lags)
    fit_seconds = perf_counter() - started
    radius = model.spectral_radius
    report = {
        "model": f"vector_ar{lags}", "lags": lags, "device": "cpu", "torch_version": str(torch.__version__), "threads": threads,
        "session": audit["session"], "input_files": audit["files"], "fit_split": "train",
        "training_raw_row_interval": audit["splits"]["train"]["raw_row_interval_half_open"],
        "training_feature_rows": len(features), "training_pairs": len(features) - lags, "features": features.shape[1],
        "sequence_length": sequence_length, "ridge": ridge, "coefficient_count": model.weight.numel() + model.bias.numel(),
        "residual_vectors": len(model.residuals), "training_contexts": len(starts),
        "training_one_step_mse_scaled": float(model.residuals.square().mean()),
        "spectral_radius": radius, "data_loading_seconds": data_seconds, "fit_seconds": fit_seconds,
        "solver": "CPU float64 centred SVD ridge solution; unpenalised intercept",
        "ridge_objective": "mean over training pairs of squared vector error + ridge * sum(weight ** 2)",
        "fit_scope": "All within-training lag histories and next-row targets, including between-window pairs and trailing rows.",
        "context_scope": "Training window starts with p previous features and matching same-state price anchors.",
        "limitations": [
            "Training MSE is an in-sample fit statistic, not validation or simulation quality.",
            "Joint empirical residuals retain simultaneous dependencies but are sampled independently over time.",
            "No price-ladder, spread or non-negative-volume constraints are enforced.",
            "A spectral radius below one describes linear mean dynamics, not validity of generated order books.",
            "Validation/test files are loaded by the existing audit, but no held-out features enter fitting or contexts.",
        ],
    }
    checkpoint = make_checkpoint(model, training, scaler, report)
    checkpoint_path, report_path = Path(checkpoint_path), Path(report_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "outputs/ar/model.pt")
    parser.add_argument("--report", type=Path, default=PROJECT_DIR / "outputs/ar/fit.json")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit_baseline(args.data_dir, args.checkpoint, args.report, args.sequence_length, args.ridge, args.threads, args.lags)
    keys = ("training_pairs", "training_contexts", "training_one_step_mse_scaled", "spectral_radius", "fit_seconds")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    print(f"Saved checkpoint: {args.checkpoint}\nSaved fit report: {args.report}")


if __name__ == "__main__":
    main()
