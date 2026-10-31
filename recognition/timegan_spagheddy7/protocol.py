"""Shared information budget for AR and noise-driven TimeGAN comparisons."""

import hashlib
import json

import numpy as np

from evaluation import conditioning_history


PRIOR_SEED_OFFSET = 10000


def comparison_protocol(sequence_length=32, max_lags=4, seeds=(42, 43, 44)):
    return {
        "id": "anchor_only_v1", "sequence_length": sequence_length, "reference_history_length": max_lags,
        "generation_seeds": list(seeds),
        "generator_heldout_inputs": [], "decoder_heldout_inputs": ["20 prices at the window's anchor snapshot"],
        "ar_initialisation": "Uniform bootstrap of training-window histories; last p rows used by VAR(p).",
        "timegan_initialisation": "Generate feature windows from noise; do not encode validation histories or futures.",
        "prior_seed_offset": PRIOR_SEED_OFFSET, "pairing": "Generated sample i is paired with chronological real window i.",
        "window_policy": "Non-overlapping sequence_length-update windows with at least reference_history_length past within-split features.",
        "selection": "Lowest seed-mean invalid rate, then mean KL, ACF error, p and ridge; SSIM is reported, not a tie-breaker.",
        "prohibited": ["held-out feature histories as generator inputs", "held-out volumes as initial state",
                       "best-of-many matching to real futures", "repairing or dropping invalid generated books"],
        "scope": "Anchored feature generation, not forecasting the particular observed future or generating starting prices.",
    }


def sample_training_prior(training, count, max_lags, seed):
    """No validation argument: held-out history cannot enter the AR initial state.

    All AR orders share the same sampled max_lags-row histories for a given
    seed; smaller orders take a suffix. Context and innovation seeds differ.
    """
    if training.raw_offset != 0:
        raise ValueError("The AR prior must come from the training split starting at raw row zero.")
    if count < 1 or max_lags < 1 or seed < 0 or seed >= 2**63 - PRIOR_SEED_OFFSET:
        raise ValueError("Invalid prior count, history length or seed.")
    pool = training.starts[training.starts >= max_lags]
    if len(pool) == 0:
        raise ValueError("No complete training windows have enough prior history.")
    rng = np.random.default_rng(seed + PRIOR_SEED_OFFSET)
    starts = pool[rng.integers(0, len(pool), size=count)]
    return conditioning_history(training, starts, max_lags), starts + training.raw_offset


def comparison_signature(protocol, metric_spec, raw_anchor_indices, input_files, scaler):
    """Record what must match in the later TimeGAN evaluation, apart from model code."""
    contract = {"protocol": protocol, "metrics": metric_spec, "anchors": list(map(int, raw_anchor_indices)),
                "input_files": input_files, "scaler": scaler}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
