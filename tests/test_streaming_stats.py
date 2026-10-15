"""Tests for streaming normalization statistics.

Covers:
- StreamingStatsAccumulator unit tests
- calculate_dataset_statistics integration tests
- calculate_delta_action_statistics integration tests
- calculate_rel_action_statistics integration tests
- Normalizer pipeline compatibility tests
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from starVLA.dataloader.gr00t_lerobot.streaming_stats import StreamingStatsAccumulator


# ---------------------------------------------------------------------------
# StreamingStatsAccumulator unit tests
# ---------------------------------------------------------------------------


class TestStreamingStatsAccumulator:

    def test_single_batch_matches_numpy(self):
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
        rng = np.random.default_rng(123)
        full_data = rng.standard_normal((10000, 7)).astype(np.float32)

        acc = StreamingStatsAccumulator()
        for start in range(0, len(full_data), 237):
            end = min(start + 237, len(full_data))
            acc.update(full_data[start:end])
        result = acc.finalize()

        np.testing.assert_allclose(result["mean"], np.mean(full_data, axis=0), rtol=1e-5)
        np.testing.assert_allclose(result["std"], np.std(full_data, axis=0), rtol=1e-4)
        np.testing.assert_allclose(result["min"], np.min(full_data, axis=0))
        np.testing.assert_allclose(result["max"], np.max(full_data, axis=0))
        np.testing.assert_allclose(result["q01"], np.percentile(full_data, 1, axis=0), rtol=0.05)
        np.testing.assert_allclose(result["q99"], np.percentile(full_data, 99, axis=0), rtol=0.05)

    def test_1d_input_reshaped(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        acc = StreamingStatsAccumulator()
        acc.update(data)
        result = acc.finalize()
        assert len(result["mean"]) == 1
        np.testing.assert_allclose(result["mean"], [3.0], rtol=1e-5)
        np.testing.assert_allclose(result["min"], [1.0])
        np.testing.assert_allclose(result["max"], [5.0])

    def test_single_element_batch(self):
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        acc = StreamingStatsAccumulator()
        acc.update(data)
        result = acc.finalize()
        np.testing.assert_allclose(result["mean"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["std"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["min"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["max"], [1.0, 2.0, 3.0])

    def test_finalize_returns_lists(self):
        acc = StreamingStatsAccumulator()
        acc.update(np.array([[1.0, 2.0]], dtype=np.float32))
        result = acc.finalize()
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            assert isinstance(result[key], list), f"{key} should be a list"
            for v in result[key]:
                assert isinstance(v, float), f"{key} elements should be float"

    def test_empty_accumulator_raises(self):
        acc = StreamingStatsAccumulator()
        with pytest.raises(ValueError, match="No data"):
            acc.finalize()

    def test_dimension_mismatch_raises(self):
        acc = StreamingStatsAccumulator()
        acc.update(np.array([[1.0, 2.0]]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            acc.update(np.array([[1.0, 2.0, 3.0]]))

    def test_zero_row_batch_ignored(self):
        acc = StreamingStatsAccumulator()
        acc.update(np.array([[1.0, 2.0]]))
        acc.update(np.empty((0, 2)))
        assert acc.count == 1
        result = acc.finalize()
        np.testing.assert_allclose(result["mean"], [1.0, 2.0])

    def test_count_property(self):
        acc = StreamingStatsAccumulator()
        assert acc.count == 0
        acc.update(np.array([[1.0], [2.0], [3.0]]))
        assert acc.count == 3
        acc.update(np.array([[4.0], [5.0]]))
        assert acc.count == 5

    def test_heavy_tailed_distribution_quantiles(self):
        rng = np.random.default_rng(99)
        data = rng.exponential(scale=2.0, size=(50000, 3)).astype(np.float32)
        acc = StreamingStatsAccumulator()
        for start in range(0, len(data), 500):
            acc.update(data[start : start + 500])
        result = acc.finalize()
        np.testing.assert_allclose(result["q01"], np.percentile(data, 1, axis=0), rtol=0.05)
        np.testing.assert_allclose(result["q99"], np.percentile(data, 99, axis=0), rtol=0.05)


# ---------------------------------------------------------------------------
# calculate_dataset_statistics integration tests
# ---------------------------------------------------------------------------


def _create_test_parquet_files(tmp_dir: Path, n_files: int = 3, rows_per_file: int = 500):
    """Create synthetic parquet files mimicking LeRobot dataset structure."""
    rng = np.random.default_rng(42)
    data_dir = tmp_dir / "data" / "chunk_000"
    data_dir.mkdir(parents=True)

    all_data = {"action.joints": [], "state.eef_position": []}
    for i in range(n_files):
        n = rows_per_file
        actions = rng.standard_normal((n, 7)).astype(np.float32)
        states = rng.standard_normal((n, 3)).astype(np.float32)

        df = pd.DataFrame({
            "action.joints": [row.tolist() for row in actions],
            "state.eef_position": [row.tolist() for row in states],
            "episode_index": np.full(n, i, dtype=np.int64),
        })
        pq.write_table(pa.Table.from_pandas(df), str(data_dir / f"episode_{i:06d}.parquet"))
        all_data["action.joints"].extend(row.tolist() for row in actions)
        all_data["state.eef_position"].extend(row.tolist() for row in states)

    return data_dir, all_data


class TestCalculateDatasetStatistics:

    def test_matches_numpy_reference(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        data_dir, all_data = _create_test_parquet_files(tmp_path)
        parquet_paths = sorted(data_dir.glob("*.parquet"))
        result = calculate_dataset_statistics(parquet_paths)

        for col_name in ("action.joints", "state.eef_position"):
            ref = np.array(all_data[col_name], dtype=np.float32)
            stats = result[col_name]
            np.testing.assert_allclose(stats["mean"], np.mean(ref, axis=0), rtol=1e-4)
            np.testing.assert_allclose(stats["std"], np.std(ref, axis=0), rtol=1e-4)
            np.testing.assert_allclose(stats["min"], np.min(ref, axis=0), atol=1e-6)
            np.testing.assert_allclose(stats["max"], np.max(ref, axis=0), atol=1e-6)
            np.testing.assert_allclose(stats["q01"], np.percentile(ref, 1, axis=0), rtol=0.05)
            np.testing.assert_allclose(stats["q99"], np.percentile(ref, 99, axis=0), rtol=0.05)

        assert "episode_index" not in result

    def test_empty_parquet_files_skipped(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        data_dir = tmp_path / "data" / "chunk_000"
        data_dir.mkdir(parents=True)
        df = pd.DataFrame({"action.x": pd.Series(dtype="object")})
        pq.write_table(pa.Table.from_pandas(df), str(data_dir / "empty.parquet"))
        result = calculate_dataset_statistics(sorted(data_dir.glob("*.parquet")))
        assert result == {}


# ---------------------------------------------------------------------------
# Delta / Rel action statistics integration tests
# ---------------------------------------------------------------------------

from starVLA.dataloader.gr00t_lerobot.schema import (
    LeRobotActionMetadata,
    LeRobotModalityField,
    LeRobotModalityMetadata,
    LeRobotStateMetadata,
)


def _create_delta_rel_test_data(tmp_dir: Path, n_episodes: int = 4, rows_per_ep: int = 50):
    rng = np.random.default_rng(777)
    data_dir = tmp_dir / "data" / "chunk_000"
    data_dir.mkdir(parents=True)

    episodes = []
    for i in range(n_episodes):
        n = rows_per_ep
        actions = rng.standard_normal((n, 3)).astype(np.float32)
        states = rng.standard_normal((n, 3)).astype(np.float32)
        df = pd.DataFrame({
            "action.pos": [row.tolist() for row in actions],
            "state.pos": [row.tolist() for row in states],
            "episode_index": np.full(n, i, dtype=np.int64),
        })
        pq.write_table(pa.Table.from_pandas(df), str(data_dir / f"episode_{i:06d}.parquet"))
        episodes.append({"action.pos": actions, "state.pos": states})

    return data_dir, episodes


def _build_simple_modality_meta():
    return LeRobotModalityMetadata(
        state={"pos": LeRobotStateMetadata(start=0, end=3, original_key="state.pos", absolute=True)},
        action={"pos": LeRobotActionMetadata(start=0, end=3, original_key="action.pos", absolute=True)},
        video={},
    )


def _compute_reference_delta_stats(episodes, action_indices, state_indices):
    all_chunks = []
    for ep in episodes:
        action_matrix, state_matrix = ep["action.pos"], ep["state.pos"]
        T = len(action_matrix)
        for base in range(T):
            chunk = np.zeros((len(action_indices), 3), dtype=np.float32)
            for ci, ai in enumerate(action_indices):
                chunk[ci] = action_matrix[max(0, min(T - 1, base + ai))]
            s_chunk = np.zeros((len(state_indices), 3), dtype=np.float32)
            for ci, si in enumerate(state_indices):
                s_chunk[ci] = state_matrix[max(0, min(T - 1, base + si))]
            out = chunk.copy()
            if len(out) > 1:
                out[1:] = chunk[1:] - chunk[:-1]
            out[0] = chunk[0] - s_chunk[0]
            all_chunks.append(out)
    all_values = np.concatenate(all_chunks, axis=0).astype(np.float32)
    return {
        "mean": np.mean(all_values, axis=0).tolist(),
        "std": np.std(all_values, axis=0).tolist(),
        "min": np.min(all_values, axis=0).tolist(),
        "max": np.max(all_values, axis=0).tolist(),
        "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
        "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
    }


def _compute_reference_rel_stats(episodes, action_indices, state_indices):
    all_chunks = []
    for ep in episodes:
        action_matrix, state_matrix = ep["action.pos"], ep["state.pos"]
        T = len(action_matrix)
        for base in range(T):
            chunk = np.zeros((len(action_indices), 3), dtype=np.float32)
            for ci, ai in enumerate(action_indices):
                chunk[ci] = action_matrix[max(0, min(T - 1, base + ai))]
            s_chunk = np.zeros((len(state_indices), 3), dtype=np.float32)
            for ci, si in enumerate(state_indices):
                s_chunk[ci] = state_matrix[max(0, min(T - 1, base + si))]
            all_chunks.append(chunk - s_chunk[0])
    all_values = np.concatenate(all_chunks, axis=0).astype(np.float32)
    return {
        "mean": np.mean(all_values, axis=0).tolist(),
        "std": np.std(all_values, axis=0).tolist(),
        "min": np.min(all_values, axis=0).tolist(),
        "max": np.max(all_values, axis=0).tolist(),
        "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
        "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
    }


class TestCalculateDeltaActionStatistics:

    def test_delta_stats_match_reference(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_delta_action_statistics

        data_dir, episodes = _create_delta_rel_test_data(tmp_path, n_episodes=10, rows_per_ep=200)
        parquet_paths = sorted(data_dir.glob("*.parquet"))
        modality_meta = _build_simple_modality_meta()
        action_indices = [0, 1, 2, 3]
        state_indices = [0]

        result = calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=modality_meta,
            action_keys_full=["action.pos"],
            state_keys_full=["state.pos"],
            action_indices=action_indices,
            state_indices=state_indices,
        )

        ref = _compute_reference_delta_stats(episodes, action_indices, state_indices)
        stats = result["action.pos"]
        np.testing.assert_allclose(stats["mean"], ref["mean"], rtol=1e-4)
        np.testing.assert_allclose(stats["std"], ref["std"], rtol=1e-4)
        np.testing.assert_allclose(stats["min"], ref["min"], atol=1e-5)
        np.testing.assert_allclose(stats["max"], ref["max"], atol=1e-5)
        np.testing.assert_allclose(stats["q01"], ref["q01"], atol=0.3)
        np.testing.assert_allclose(stats["q99"], ref["q99"], atol=0.3)

    def test_delta_stats_single_step_horizon(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_delta_action_statistics

        data_dir, episodes = _create_delta_rel_test_data(tmp_path, n_episodes=2, rows_per_ep=20)
        parquet_paths = sorted(data_dir.glob("*.parquet"))
        result = calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=_build_simple_modality_meta(),
            action_keys_full=["action.pos"],
            state_keys_full=["state.pos"],
            action_indices=[0],
            state_indices=[0],
        )
        ref = _compute_reference_delta_stats(episodes, [0], [0])
        np.testing.assert_allclose(result["action.pos"]["mean"], ref["mean"], rtol=1e-4)
        np.testing.assert_allclose(result["action.pos"]["std"], ref["std"], rtol=1e-4)


class TestCalculateRelActionStatistics:

    def test_rel_stats_match_reference(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_rel_action_statistics

        data_dir, episodes = _create_delta_rel_test_data(tmp_path, n_episodes=10, rows_per_ep=200)
        parquet_paths = sorted(data_dir.glob("*.parquet"))
        result = calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=_build_simple_modality_meta(),
            action_keys_full=["action.pos"],
            state_keys_full=["state.pos"],
            action_indices=[0, 1, 2, 3],
            state_indices=[0],
        )
        ref = _compute_reference_rel_stats(episodes, [0, 1, 2, 3], [0])
        stats = result["action.pos"]
        np.testing.assert_allclose(stats["mean"], ref["mean"], rtol=1e-4)
        np.testing.assert_allclose(stats["std"], ref["std"], rtol=1e-4)
        np.testing.assert_allclose(stats["min"], ref["min"], atol=1e-5)
        np.testing.assert_allclose(stats["max"], ref["max"], atol=1e-5)
        np.testing.assert_allclose(stats["q01"], ref["q01"], atol=0.3)
        np.testing.assert_allclose(stats["q99"], ref["q99"], atol=0.3)

    def test_rel_stats_preserves_base_stats(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_rel_action_statistics

        data_dir, _ = _create_delta_rel_test_data(tmp_path, n_episodes=2, rows_per_ep=20)
        result = calculate_rel_action_statistics(
            parquet_paths=sorted((data_dir).glob("*.parquet")),
            lerobot_modality_meta=_build_simple_modality_meta(),
            action_keys_full=["action.pos"],
            state_keys_full=["state.pos"],
            action_indices=[0],
            state_indices=[0],
        )
        assert "state.pos" in result
        assert "mean" in result["state.pos"]


# ---------------------------------------------------------------------------
# Normalizer pipeline compatibility tests
# ---------------------------------------------------------------------------


class TestNormalizerPipeline:

    def test_q99_normalize_and_inverse(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        data_dir, all_data = _create_test_parquet_files(tmp_path)
        stats = calculate_dataset_statistics(sorted(data_dir.glob("*.parquet")))
        feat_stats = stats["action.joints"]

        q01 = torch.tensor(feat_stats["q01"])
        q99 = torch.tensor(feat_stats["q99"])
        raw_data = torch.tensor(all_data["action.joints"][:16], dtype=torch.float32)

        # Forward normalization
        mask = q01 != q99
        normalized = torch.zeros_like(raw_data)
        normalized[..., mask] = (raw_data[..., mask] - q01[mask]) / (q99[mask] - q01[mask])
        normalized[..., mask] = 2 * normalized[..., mask] - 1
        normalized = torch.clamp(normalized, -1, 1)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

        # Inverse
        recovered = (normalized + 1) / 2 * (q99 - q01) + q01

        for dim in range(raw_data.shape[1]):
            if not mask[dim]:
                continue
            clamped = raw_data[:, dim].clamp(q01[dim].item(), q99[dim].item())
            torch.testing.assert_close(recovered[:, dim], clamped, rtol=1e-5, atol=1e-5)

    def test_stats_json_serializable(self, tmp_path):
        from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

        data_dir, _ = _create_test_parquet_files(tmp_path)
        stats = calculate_dataset_statistics(sorted(data_dir.glob("*.parquet")))
        json_str = json.dumps(stats)
        roundtrip = json.loads(json_str)
        for feat in stats:
            for key in ("mean", "std", "min", "max", "q01", "q99"):
                assert roundtrip[feat][key] == stats[feat][key]
