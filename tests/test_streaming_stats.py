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


import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _create_test_parquet_files(tmp_dir: Path, n_files: int = 3, rows_per_file: int = 500):
    """Create synthetic parquet files mimicking LeRobot dataset structure."""
    rng = np.random.default_rng(42)
    data_dir = tmp_dir / "data" / "chunk_000"
    data_dir.mkdir(parents=True)

    all_data = {"action.joints": [], "state.eef_position": [], "episode_index": []}
    for i in range(n_files):
        n = rows_per_file
        actions = rng.standard_normal((n, 7)).astype(np.float32)
        states = rng.standard_normal((n, 3)).astype(np.float32)
        episodes = np.full(n, i, dtype=np.int64)

        df = pd.DataFrame(
            {
                "action.joints": [row.tolist() for row in actions],
                "state.eef_position": [row.tolist() for row in states],
                "episode_index": episodes,
            }
        )
        pq.write_table(pa.Table.from_pandas(df), str(data_dir / f"episode_{i:06d}.parquet"))

        all_data["action.joints"].extend(row.tolist() for row in actions)
        all_data["state.eef_position"].extend(row.tolist() for row in states)

    return data_dir, all_data


class TestCalculateDatasetStatistics:
    """Integration test: streaming stats match numpy reference."""

    def test_matches_numpy_reference(self, tmp_path):
        data_dir, all_data = _create_test_parquet_files(tmp_path)
        parquet_paths = sorted(data_dir.glob("*.parquet"))

        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        result = calculate_dataset_statistics(parquet_paths)

        for col_name in ("action.joints", "state.eef_position"):
            ref = np.array(all_data[col_name], dtype=np.float32)
            stats = result[col_name]

            np.testing.assert_allclose(stats["mean"], np.mean(ref, axis=0), rtol=1e-4)
            np.testing.assert_allclose(stats["std"], np.std(ref, axis=0), rtol=1e-4)
            np.testing.assert_allclose(stats["min"], np.min(ref, axis=0), atol=1e-6)
            np.testing.assert_allclose(stats["max"], np.max(ref, axis=0), atol=1e-6)
            np.testing.assert_allclose(
                stats["q01"], np.percentile(ref, 1, axis=0), rtol=0.05
            )
            np.testing.assert_allclose(
                stats["q99"], np.percentile(ref, 99, axis=0), rtol=0.05
            )

        # episode_index should be excluded (metadata column)
        assert "episode_index" not in result

    def test_empty_parquet_files_skipped(self, tmp_path):
        data_dir = tmp_path / "data" / "chunk_000"
        data_dir.mkdir(parents=True)
        # Create an empty parquet file
        df = pd.DataFrame({"action.x": pd.Series(dtype="object")})
        pq.write_table(pa.Table.from_pandas(df), str(data_dir / "empty.parquet"))

        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        result = calculate_dataset_statistics(sorted(data_dir.glob("*.parquet")))
        assert result == {}
