"""Guard validation metrics, selection and training-only comparison inputs."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataset import PRICE_COLUMNS, FeatureScaler
from evaluation import (absolute_return_acf, conditioning_history, decode_batch, histogram_kl, histogram_probabilities,
                        market_series, validation_windows, validity_rates, DepthSSIM, depth_images, depth_ssim_specification)
from tune_ar import select_candidate
from protocol import comparison_protocol, comparison_signature, sample_training_prior


class EvaluationChecks(unittest.TestCase):
    def test_depth_ssim_matches_independent_patch_statistics(self):
        rng = np.random.default_rng(17)
        real, generated = rng.uniform(1, 500, (2, 10, 40)), rng.uniform(1, 500, (2, 10, 40))
        spec = depth_ssim_specification(SimpleNamespace(book=real[0]))
        x, y = depth_images(real, spec), depth_images(generated, spec)
        expected = np.empty((2, 2))
        for batch in range(2):
            for side in range(2):
                patches = []
                for i in range(4):
                    for j in range(4):
                        a, b = x[batch, side, i:i + 7, j:j + 7].ravel(), y[batch, side, i:i + 7, j:j + 7].ravel()
                        covariance = np.cov(a, b, ddof=1)
                        luminance = (2 * a.mean() * b.mean() + .01 ** 2) / (a.mean() ** 2 + b.mean() ** 2 + .01 ** 2)
                        structure = (2 * covariance[0, 1] + .03 ** 2) / (covariance.trace() + .03 ** 2)
                        patches.append(luminance * structure)
                expected[batch, side] = np.mean(patches)
        scores, sides = DepthSSIM(real, spec).score(generated)
        np.testing.assert_allclose(sides, expected, atol=1e-12)
        np.testing.assert_allclose(scores, expected.mean(axis=1), atol=1e-12)

    def test_depth_identity_prices_and_invalid_volume_are_not_conflated(self):
        real = np.full((2, 32, 40), 100.)
        spec = depth_ssim_specification(SimpleNamespace(book=real[0]))
        evaluator = DepthSSIM(real, spec)
        invalid = real.copy()
        invalid[..., 0] = -1  # Price validity is outside this volume-only metric.
        np.testing.assert_allclose(evaluator.score(invalid)[0], 1, atol=1e-12)
        invalid[..., 1::2] = -100
        image = depth_images(invalid, spec)
        np.testing.assert_allclose(image, -1)
        self.assertTrue((evaluator.score(invalid)[0] < 0).all())
        invalid[..., 1::2] = 10000
        self.assertTrue((depth_images(invalid, spec) > 1).all())
        self.assertAlmostEqual(spec["training_log_volume_max"], np.log1p(100))
        np.testing.assert_array_equal(real, 100)
        with self.assertRaises(ValueError):
            evaluator.score(invalid[:1])

    def test_depth_ssim_constant_images_have_analytic_luminance_score(self):
        real, generated = np.full((1, 32, 40), 9.), np.full((1, 32, 40), 99.)
        spec = depth_ssim_specification(SimpleNamespace(book=generated[0]))
        # log(10)/log(100) = 0.5, so all variance and covariance terms vanish.
        expected = (2 * .5 + .01 ** 2) / (.5 ** 2 + 1 + .01 ** 2)
        np.testing.assert_allclose(DepthSSIM(real, spec).score(generated)[0], expected, atol=1e-12)

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


class ProtocolChecks(unittest.TestCase):
    def test_prior_repeats_without_touching_global_rng_or_heldout_data(self):
        features = np.arange(128 * 40).reshape(128, 40).astype(float)
        training = SimpleNamespace(raw_offset=0, starts=np.array([0, 32, 64, 96]), features=features)
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        histories, rows = sample_training_prior(training, 30, 4, 42)
        repeated, same_rows = sample_training_prior(training, 30, 4, 42)
        torch.testing.assert_close(histories, repeated)
        np.testing.assert_array_equal(rows, same_rows)
        for index, row in enumerate(rows):
            np.testing.assert_array_equal(histories[index], features[row - 4:row])
            np.testing.assert_array_equal(histories[index, -1:], features[row - 1:row])
        self.assertTrue(set(rows).issubset({32, 64, 96}))
        self.assertFalse(np.array_equal(rows, sample_training_prior(training, 30, 4, 43)[1]))
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        with self.assertRaises(ValueError):
            sample_training_prior(SimpleNamespace(**{**vars(training), "raw_offset": 128}), 30, 4, 42)

    def test_contract_changes_when_information_or_comparison_changes(self):
        protocol = comparison_protocol()
        self.assertEqual(protocol["generator_heldout_inputs"], [])
        self.assertEqual(protocol["generation_seeds"], [42, 43, 44])
        args = (protocol, {"ssim": {"data_range": 1}}, [32, 64], [{"sha256": "dataset hash"}], {"minimum": [0]})
        signature = comparison_signature(*args)
        self.assertEqual(signature, comparison_signature(*args))
        self.assertNotEqual(signature, comparison_signature(args[0], args[1], [64, 32], *args[3:]))
        self.assertNotEqual(signature, comparison_signature({**protocol, "generator_heldout_inputs": ["history"]}, *args[1:]))
        self.assertNotEqual(signature, comparison_signature(args[0], {"ssim": {"data_range": 2}}, *args[2:]))


if __name__ == "__main__":
    unittest.main()
