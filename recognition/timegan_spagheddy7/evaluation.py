"""Shared validation diagnostics; no fitting, clipping or structural repair."""

import numpy as np
import torch

from dataset import PRICE_COLUMNS, PRICE_DIVISOR, VOLUME_COLUMNS, decode_features


def validation_windows(dataset, max_lags):
    """Use common starts, with all observed conditioning history inside this split."""
    starts = dataset.starts[dataset.starts >= max_lags]
    if len(starts) == 0:
        raise ValueError("No complete windows have enough within-split history.")
    anchors = dataset.book[starts][:, PRICE_COLUMNS].astype(np.float64) / PRICE_DIVISOR
    real = np.stack([dataset.book[s + 1:s + dataset.sequence_length + 1] for s in starts]).astype(np.float64)
    real[..., PRICE_COLUMNS] /= PRICE_DIVISOR
    return starts, anchors, real


def conditioning_history(dataset, starts, lags):
    indices = starts[:, None] + np.arange(-lags, 0)
    if np.any(indices < 0):
        raise ValueError("Conditioning history would cross a split boundary.")
    return torch.as_tensor(dataset.features[indices], dtype=torch.float64)


def decode_batch(features, scaler, anchors):
    values = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    if len(values) != len(anchors):
        raise ValueError("Each generated sequence requires its own price anchor.")
    with np.errstate(over="raise", invalid="raise"):
        books = decode_features(values, scaler, anchors)
    if not np.isfinite(books).all():
        raise FloatingPointError("Non-finite decoded books; no repair or sample filtering applied.")
    return books


def validity_rates(books):
    """Strict violations per snapshot; categories overlap and include all ten levels."""
    asks, bids, volumes = books[..., 0::4], books[..., 2::4], books[..., VOLUME_COLUMNS]
    flags = {
        "nonfinite_rate": ~np.isfinite(books).all(axis=-1),
        "nonpositive_price_rate": np.any(books[..., PRICE_COLUMNS] <= 0, axis=-1),
        "crossed_rate": asks[..., 0] <= bids[..., 0],
        "ask_order_rate": np.any(np.diff(asks, axis=-1) <= 0, axis=-1),
        "bid_order_rate": np.any(np.diff(bids, axis=-1) >= 0, axis=-1),
        "negative_volume_rate": np.any(volumes < 0, axis=-1),
    }
    flags["invalid_rate"] = np.logical_or.reduce(list(flags.values()))
    return {name: float(flag.mean()) for name, flag in flags.items()}


def market_series(books, anchors):
    spread = books[..., 0] - books[..., 2]
    mid = (books[..., 0] + books[..., 2]) / 2
    initial_mid = (anchors[:, 0] + anchors[:, 1]) / 2
    if not np.isfinite(books).all() or np.any(mid <= 0) or np.any(initial_mid <= 0):
        raise FloatingPointError("Distribution metrics require finite books and positive mid-prices.")
    returns = np.diff(np.log(np.column_stack((initial_mid, mid))), axis=1)
    return {"spread": spread, "mid_return": returns}


def metric_specification(training, bins=50, max_lag=10, smoothing=1e-8):
    """Freeze bin boundaries from training only, with separate infinite tail bins."""
    book = training.book.astype(np.float64)
    spreads = (book[:, 0] - book[:, 2]) / PRICE_DIVISOR
    mid = (book[:, 0] + book[:, 2]) / (2 * PRICE_DIVISOR)
    series = {"spread": spreads, "mid_return": np.diff(np.log(mid))}
    boundaries = {}
    for name, values in series.items():
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        boundaries[name] = np.linspace(low, high, bins + 1).tolist()
    return {"boundaries": boundaries, "regular_bins": bins, "tail_bins": 2, "smoothing": smoothing,
            "max_lag": max_lag, "kl_direction": "real || generated", "logarithm": "natural",
            "bin_source": "complete training segment only", "structural_repair": "none"}


def histogram_probabilities(values, boundaries, smoothing):
    if not np.isfinite(values).all():
        raise FloatingPointError("Histogram inputs must be finite; do not silently drop invalid samples.")
    counts = np.histogram(np.asarray(values).ravel(), bins=np.r_[-np.inf, boundaries, np.inf])[0]
    probabilities = counts / counts.sum() + smoothing
    return probabilities / probabilities.sum()


def histogram_kl(real, generated, boundaries, smoothing=1e-8):
    p = histogram_probabilities(real, boundaries, smoothing)
    q = histogram_probabilities(generated, boundaries, smoothing)
    return float(np.sum(p * np.log(p / q)))


def absolute_return_acf(returns, max_lag=10):
    """Pooled, globally centred ACF with no pairs crossing 32-update windows.

    Divide each lag's cross-product sum by the full sum of squared deviations.
    This biased estimate is bounded; constant absolute returns give zero ACF.
    """
    if returns.ndim != 2 or max_lag >= returns.shape[1] or max_lag < 1:
        raise ValueError("Need (windows, updates) returns and 1 <= max_lag < updates.")
    values = np.abs(returns)
    centred = values - values.mean()
    denominator = np.sum(centred ** 2)
    if denominator == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(centred[:, :-lag] * centred[:, lag:]) / denominator for lag in range(1, max_lag + 1)])


def evaluate_books(real, generated, anchors, spec):
    return ValidationEvaluator(real, anchors, spec).evaluate(generated)[0]


class ValidationEvaluator:
    """Cache real reference statistics once for a shared validation window set."""

    def __init__(self, real, anchors, spec):
        self.shape, self.anchors, self.spec = real.shape, anchors, spec
        actual = market_series(real, anchors)
        self.probabilities = self.histograms(actual)
        self.acf = absolute_return_acf(actual["mid_return"], spec["max_lag"])

    def histograms(self, series):
        return {name: histogram_probabilities(values, self.spec["boundaries"][name], self.spec["smoothing"])
                for name, values in series.items()}

    def evaluate(self, generated):
        if generated.shape != self.shape or generated.ndim != 3 or generated.shape[-1] != 40:
            raise ValueError("Real and generated books must have identical (windows, updates, 40) shapes.")
        simulated = market_series(generated, self.anchors)
        probabilities = self.histograms(simulated)
        acf = absolute_return_acf(simulated["mid_return"], self.spec["max_lag"])
        scores = validity_rates(generated)
        for name, p in self.probabilities.items():
            scores[f"kl_{name}"] = float(np.sum(p * np.log(p / probabilities[name])))
        scores["abs_return_acf_mae"] = float(np.abs(self.acf - acf).mean())
        scores["mean_kl"] = (scores["kl_spread"] + scores["kl_mid_return"]) / 2
        return scores, {"acf": acf, **probabilities}
