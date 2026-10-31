"""Reload the AR baseline and generate training-context examples, without CSV access."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FEATURE_NAMES, FeatureScaler, PROJECT_DIR, sha256_file
from evaluation import decode_batch
from modules import VectorAR


def save_preview(book, initial_prices, destination, lags=1):
    """Show the first generated sequence without filtering or repairing it."""
    prices = np.vstack((initial_prices[[0, 1]], book[:, [0, 2]]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    axes[0].plot(np.arange(len(prices)), prices[:, 0], label="Best ask")
    axes[0].plot(np.arange(len(prices)), prices[:, 1], label="Best bid")
    axes[0].set(xlabel="Generated update (0 = supplied anchor)", ylabel="Price (USD)", title="Generated best prices")
    axes[1].plot(np.arange(1, len(book) + 1), book[:, 1], label="Best ask volume")
    axes[1].plot(np.arange(1, len(book) + 1), book[:, 3], label="Best bid volume")
    axes[1].set(xlabel="Generated update", ylabel="Shares", title="Generated level-1 volumes")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(f"VAR({lags}): first generated example; training context, no structural repair")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def generate_examples(checkpoint_path, output_path, report_path, figure_path, count=64, seed=42, threads=4):
    if count < 1 or seed < 0 or seed >= 2**63 - 1 or threads < 1:
        raise ValueError("Use a positive count/threads and a seed between 0 and 2**63 - 2.")
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    supported = {(1, "vector_ar1"), (2, "vector_ar")}
    if (checkpoint.get("format_version"), checkpoint.get("model_type")) not in supported:
        raise ValueError("Unsupported checkpoint format or model type.")
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError("Checkpoint features do not match the current data pipeline.")
    model = VectorAR(**checkpoint["state_dict"]).eval()
    scaler = FeatureScaler(checkpoint["scaler"]["minimum"].numpy(), checkpoint["scaler"]["maximum"].numpy())
    contexts = checkpoint["contexts"]
    rng = torch.Generator().manual_seed(seed)
    selected = torch.randint(len(contexts["features"]), (count,), generator=rng)
    initial_features, initial_prices = contexts["features"][selected], contexts["initial_prices"][selected]
    sequence_length = checkpoint["fit_report"]["sequence_length"]

    started = perf_counter()
    # Separate deterministic streams for choosing contexts and innovations.
    features = model.generate(initial_features, sequence_length=sequence_length, seed=seed + 1)
    generation_seconds = perf_counter() - started
    books = decode_batch(features, scaler, initial_prices.numpy())

    output_path, report_path = Path(output_path), Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, scaled_features=features.numpy(), orderbooks=books,
                        initial_features=initial_features.numpy(), initial_prices=initial_prices.numpy(),
                        context_raw_start_indices=contexts["raw_start_indices"][selected].numpy())
    save_preview(books[0], initial_prices[0].numpy(), figure_path, model.lags)
    report = {
        "model": f"vector_ar{model.lags}", "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": threads,
        "context_split": "train", "context_seed": seed, "innovation_seed": seed + 1,
        "sequences": count, "sequence_length": sequence_length, "features": len(FEATURE_NAMES),
        "generated_shape": list(features.shape), "all_features_finite": True, "all_decoded_values_finite": True,
        "generation_seconds": generation_seconds, "sequences_per_second": count / generation_seconds,
        "timing_scope": "Feature rollout only; excludes model loading, context selection, decoding, file IO and plotting.",
        "first_context_raw_start_index": int(contexts["raw_start_indices"][selected[0]]),
        "initial_state_policy": "Previous p features and matching price anchor sampled from training; all returned updates generated.",
        "structural_repair": "none",
        "status": "Fit/save/reload/generate smoke test only; no held-out quality evaluation.",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "outputs/ar/model.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs/ar/generated.npz")
    parser.add_argument("--report", type=Path, default=PROJECT_DIR / "outputs/ar/generation.json")
    parser.add_argument("--figure", type=Path, default=PROJECT_DIR / "outputs/ar/preview.png")
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = generate_examples(args.checkpoint, args.output, args.report, args.figure,
                               args.num_sequences, args.seed, args.threads)
    print(json.dumps(report, indent=2))
    print(f"Saved generated arrays: {args.output}\nSaved preview: {args.figure}")


if __name__ == "__main__":
    main()
