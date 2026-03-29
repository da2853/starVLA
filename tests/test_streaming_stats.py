"""Tests for StreamingStatsAccumulator."""

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.streaming_stats import StreamingStatsAccumulator


class TestStreamingStatsAccumulator:
    """Unit tests for the streaming accumulator."""

    def test_single_batch_matches_numpy(self):
        """A single batch should produce identical results to numpy."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 4)).astype(np.float32)

        acc = StreamingStatsAccumulator()
        acc.update(data)
        result = acc.finalize()

        np.testing.assert_allclose(result["mean"], np.mean(data, axis=0), rtol=1e-5)
        np.testing.assert_allclose(result["std"], np.std(data, axis=0), rtol=1e-5)
        np.testing.assert_allclose(result["min"], np.min(data, axis=0))
        np.testing.assert_allclose(result["max"], np.max(data, axis=0))
        np.testing.assert_allclose(result["q01"], np.percentile(data, 1, axis=0), rtol=0.05)
        np.testing.assert_allclose(result["q99"], np.percentile(data, 99, axis=0), rtol=0.05)

    def test_multiple_batches_matches_numpy(self):
        """Streaming across multiple batches should match computing on full data."""
        rng = np.random.default_rng(123)
        full_data = rng.standard_normal((10000, 7)).astype(np.float32)

        acc = StreamingStatsAccumulator()
        # Feed in chunks of varying sizes
        for start in range(0, len(full_data), 237):
            end = min(start + 237, len(full_data))
            acc.update(full_data[start:end])
        result = acc.finalize()

        np.testing.assert_allclose(result["mean"], np.mean(full_data, axis=0), rtol=1e-5)
        np.testing.assert_allclose(result["std"], np.std(full_data, axis=0), rtol=1e-4)
        np.testing.assert_allclose(result["min"], np.min(full_data, axis=0))
        np.testing.assert_allclose(result["max"], np.max(full_data, axis=0))
        np.testing.assert_allclose(
            result["q01"], np.percentile(full_data, 1, axis=0), rtol=0.05
        )
        np.testing.assert_allclose(
            result["q99"], np.percentile(full_data, 99, axis=0), rtol=0.05
        )

    def test_1d_input_reshaped(self):
        """1D input should be treated as [N, 1]."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        acc = StreamingStatsAccumulator()
        acc.update(data)
        result = acc.finalize()

        assert len(result["mean"]) == 1
        np.testing.assert_allclose(result["mean"], [3.0], rtol=1e-5)
        np.testing.assert_allclose(result["min"], [1.0])
        np.testing.assert_allclose(result["max"], [5.0])

    def test_single_element_batch(self):
        """A batch with one row should not crash or produce NaN."""
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        acc = StreamingStatsAccumulator()
        acc.update(data)
        result = acc.finalize()

        np.testing.assert_allclose(result["mean"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["std"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["min"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["max"], [1.0, 2.0, 3.0])

    def test_finalize_returns_lists(self):
        """finalize() should return Python lists, not numpy arrays."""
        acc = StreamingStatsAccumulator()
        acc.update(np.array([[1.0, 2.0]], dtype=np.float32))
        result = acc.finalize()

        for key in ("mean", "std", "min", "max", "q01", "q99"):
            assert isinstance(result[key], list), f"{key} should be a list"
            for v in result[key]:
                assert isinstance(v, float), f"{key} elements should be float"

    def test_empty_accumulator_raises(self):
        """finalize() on a never-updated accumulator should raise."""
        acc = StreamingStatsAccumulator()
        with pytest.raises(ValueError, match="No data"):
            acc.finalize()

    def test_dimension_mismatch_raises(self):
        """Feeding batches with different dimensions should raise."""
        acc = StreamingStatsAccumulator()
        acc.update(np.array([[1.0, 2.0]]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            acc.update(np.array([[1.0, 2.0, 3.0]]))

    def test_heavy_tailed_distribution_quantiles(self):
        """Quantiles should be accurate even for non-Gaussian data."""
        rng = np.random.default_rng(99)
        # Exponential distribution: heavy right tail
        data = rng.exponential(scale=2.0, size=(50000, 3)).astype(np.float32)

        acc = StreamingStatsAccumulator()
        for start in range(0, len(data), 500):
            acc.update(data[start : start + 500])
        result = acc.finalize()

        np.testing.assert_allclose(
            result["q01"], np.percentile(data, 1, axis=0), rtol=0.05
        )
        np.testing.assert_allclose(
            result["q99"], np.percentile(data, 99, axis=0), rtol=0.05
        )
