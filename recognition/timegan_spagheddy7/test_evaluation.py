"""Guard validation alignment, strict validity, tail bins and selection semantics."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataset import PRICE_COLUMNS, FeatureScaler
from evaluation import (absolute_return_acf, conditioning_history, decode_batch, histogram_kl, histogram_probabilities,
                        market_series, validation_windows, validity_rates)
from tune_ar import select_candidate


class EvaluationChecks(unittest.TestCase):
    def test_batch_decoding_preserves_anchors_returns_and_invalid_volumes(self):
        features = np.zeros((2, 3, 40))
        features[0, :, PRICE_COLUMNS] = np.log(1.1)
        features[1, :, PRICE_COLUMNS] = np.log(0.9)
        features[..., 1::2] = np.log(5)
        features[1, 2, 39] = np.log(0.5)
        anchors = np.stack((np.full(20, 100.), np.full(20, 200.)))
        result = decode_batch(features, FeatureScaler(np.zeros(40), np.ones(40)), anchors)
        np.testing.assert_allclose(result[0, :, 0], 100 * 1.1 ** np.arange(1, 4))
        np.testing.assert_allclose(result[1, :, 0], 200 * 0.9 ** np.arange(1, 4))
        self.assertAlmostEqual(result[0, 0, 1], 4)
        self.assertAlmostEqual(result[1, 2, 39], -0.5)

    def test_validation_history_anchor_and_future_are_aligned(self):
        book = np.arange(11 * 40).reshape(11, 40) + 10000
        features = np.arange(10 * 40).reshape(10, 40).astype(np.float32)
        split = SimpleNamespace(book=book, features=features, starts=np.array([0, 3, 6]), sequence_length=3)
        starts, anchors, real = validation_windows(split, max_lags=2)
        np.testing.assert_array_equal(starts, [3, 6])
        torch.testing.assert_close(conditioning_history(split, starts, 2)[0], torch.tensor(features[1:3], dtype=torch.float64))
        np.testing.assert_array_equal(anchors[0], book[3, PRICE_COLUMNS] / 10000)
        np.testing.assert_array_equal(real[0, :, 0], book[4:7, 0] / 10000)
        with self.assertRaises(ValueError):
            conditioning_history(split, np.array([0]), 1)

    def test_validity_checks_all_levels_and_counts_union(self):
        books = np.ones((1, 3, 40))
        books[..., 0::4], books[..., 2::4] = 101 + np.arange(10), 99 - np.arange(10)
        self.assertEqual(validity_rates(books)["invalid_rate"], 0)
        books[0, 0, 36] = books[0, 0, 32]  # Equality at deepest ask level is invalid.
        books[0, 1, 0] = books[0, 1, 2]  # Zero best spread.
        books[0, 1, 39] = -1  # Overlap with the preceding violation.
        rates = validity_rates(books)
        self.assertEqual(rates["invalid_rate"], 2 / 3)
        self.assertEqual(rates["ask_order_rate"], 1 / 3)
        self.assertEqual(rates["crossed_rate"], 1 / 3)
        self.assertEqual(rates["negative_volume_rate"], 1 / 3)

    def test_histogram_includes_both_out_of_range_tails(self):
        probabilities = histogram_probabilities(np.array([-100., 0.5, 100.]), [0., 1.], smoothing=0)
        np.testing.assert_allclose(probabilities, [1 / 3, 1 / 3, 1 / 3])
        real = np.array([0.2, 0.5, 0.8])
        self.assertAlmostEqual(histogram_kl(real, real, [0., 1.]), 0)
        self.assertGreater(histogram_kl(real, real + 100, [0., 1.]), 10)
        with self.assertRaises(FloatingPointError):
            histogram_probabilities(np.array([np.nan]), [0., 1.], smoothing=1e-8)

    def test_acf_does_not_join_independent_window_boundaries(self):
        returns = np.array([[1., 1., 1.], [3., 3., 3.]])
        np.testing.assert_allclose(absolute_return_acf(returns, max_lag=2), [2 / 3, 1 / 3])
        np.testing.assert_array_equal(absolute_return_acf(np.ones((2, 3)), 2), [0, 0])

    def test_first_mid_return_uses_its_own_anchor(self):
        books = np.ones((2, 2, 40))
        books[0, :, [0, 2]], books[1, :, [0, 2]] = 110, 180
        anchors = np.ones((2, 20))
        anchors[0], anchors[1] = 100, 200
        returns = market_series(books, anchors)["mid_return"]
        np.testing.assert_allclose(returns[:, 0], np.log([1.1, 0.9]))
        np.testing.assert_array_equal(returns[:, 1], [0, 0])

    def test_selection_prioritises_validity_and_excludes_failed_candidates(self):
        base = {"status": "ok", "lags": 1, "ridge": 1e-4, "abs_return_acf_mae_mean": 0.1}
        lower_kl = dict(base, candidate="low_kl", invalid_rate_mean=0.5, mean_kl_mean=0.01)
        lower_invalid = dict(base, candidate="low_invalid", invalid_rate_mean=0.4, mean_kl_mean=1)
        failed = dict(base, candidate="failed", status="failed", invalid_rate_mean=0, mean_kl_mean=0)
        self.assertEqual(select_candidate([failed, lower_kl, lower_invalid])["candidate"], "low_invalid")


if __name__ == "__main__":
    unittest.main()
