# tests/test_datagen_utils.py
"""Tests for sims.data_generation.datagen_utils module."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from sims.data_generation.datagen_utils import (
    build_generation_config,
    split_house_repeats_to_id,
    parse_queue_message,
    skip_keys,
    write_config_file,
)


class _ActionSpaceA:
    pass


class _ActionSpaceB:
    pass


def _generation_args(**overrides):
    values = {
        "max_steps": 1000,
        "trajectories_per_house": 2,
        "house_dataset": "objaverse",
        "resolution_scale": 0.5,
        "quality": "Ultra",
        "material_randomization_probability": 0.8,
        "rotation_noise_std_degrees": 0.5,
        "max_houses": 100,
        "workers": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_config(
    args,
    action_space,
    video_modalities=("rgb",),
):
    return build_generation_config(
        args,
        action_space,
        task_type="HouseWalkthrough",
        width=198,
        height=112,
        video_modalities=video_modalities,
    )


def test_generation_config_captures_semantics_not_scheduling():
    config = _build_config(_generation_args(), _ActionSpaceA())
    values = OmegaConf.to_container(config)

    assert values["schema_version"] == 4
    assert values["task"] == {
        "type": "HouseWalkthrough",
        "max_steps": 1000,
        "trajectories_per_house": 2,
    }
    assert values["source"] == {"house_dataset": "objaverse"}
    assert values["action_space"]["id"] == "discrete-stretch-v1"
    assert values["rendering"] == {
        "resolution_scale": 0.5,
        "controller_width": 198,
        "controller_height": 112,
        "raw_camera_width": 186,
        "raw_camera_height": 112,
        "quality": "Ultra",
        "video_modalities": ["rgb"],
    }
    assert values["randomization"] == {
        "material_probability": 0.8,
        "rotation_noise_std_degrees": 0.5,
    }
    assert "n_workers" not in json.dumps(values)
    assert "max_houses" not in json.dumps(values)


def test_write_config_allows_same_config_and_rejects_mismatch(tmp_path):
    action_space = _ActionSpaceA()
    config = _build_config(_generation_args(), action_space)

    write_config_file(config, str(tmp_path))
    write_config_file(config, str(tmp_path))

    assert (tmp_path / "constants.yaml").is_file()
    assert not (tmp_path / "action_space.pkl").exists()

    mismatched = _build_config(_generation_args(quality="Low"), action_space)
    with pytest.raises(ValueError, match="does not match the existing dataset"):
        write_config_file(mismatched, str(tmp_path))

    different_modalities = _build_config(
        _generation_args(),
        action_space,
        video_modalities=("rgb", "depth"),
    )
    with pytest.raises(ValueError, match="does not match the existing dataset"):
        write_config_file(different_modalities, str(tmp_path))


def test_write_config_ignores_legacy_action_space_pickle(tmp_path):
    action_space = _ActionSpaceA()
    config = _build_config(_generation_args(), action_space)
    write_config_file(config, str(tmp_path))

    legacy_pickle = tmp_path / "action_space.pkl"
    legacy_pickle.write_bytes(b"an unreadable legacy pickle")

    write_config_file(config, str(tmp_path))

    assert legacy_pickle.read_bytes() == b"an unreadable legacy pickle"


def test_write_config_rejects_action_space_mismatch_from_constants(tmp_path):
    saved_config = _build_config(_generation_args(), _ActionSpaceA())
    write_config_file(saved_config, str(tmp_path))

    current_config = _build_config(_generation_args(), _ActionSpaceB())
    with pytest.raises(ValueError, match="does not match the existing dataset"):
        write_config_file(current_config, str(tmp_path))


class TestSplitHouseRepeatsToId:
    """Test split_house_repeats_to_id function."""

    def test_basic_formatting(self):
        """Test basic ID formatting."""
        result = split_house_repeats_to_id("train", 1, 2)
        assert result == "split_train__house_00000001__repeats_02"

    def test_large_house_number(self):
        """Test formatting with large house number."""
        result = split_house_repeats_to_id("val", 12345678, 5)
        assert result == "split_val__house_12345678__repeats_05"

    def test_zero_values(self):
        """Test formatting with zero values."""
        result = split_house_repeats_to_id("test", 0, 0)
        assert result == "split_test__house_00000000__repeats_00"

    @pytest.mark.parametrize(
        "split,house,repeats",
        [
            ("train", 100, 10),
            ("val", 999, 99),
            ("test", 1, 1),
            ("train_unseen", 50, 5),
        ],
    )
    def test_various_inputs(self, split, house, repeats):
        """Test with various input combinations."""
        result = split_house_repeats_to_id(split, house, repeats)
        assert f"split_{split}" in result
        assert f"house_{house:08d}" in result
        assert f"repeats_{repeats:02d}" in result


class TestParseQueueMessage:
    """Test parse_queue_message function."""

    def _make_message(self, body: str) -> MagicMock:
        """Create a mock message with given body."""
        msg = MagicMock()
        msg.body = body
        return msg

    def test_parse_valid_message(self):
        """Test parsing a valid message."""
        msg = self._make_message("split_train__house_00000001__repeats_02")
        result = parse_queue_message(msg, "train")

        assert result["id"] == "split_train__house_00000001__repeats_02"
        assert result["split"] == "train"
        assert result["house_index"] == 1
        assert result["repeats"] == 2

    def test_parse_val_split(self):
        """Test parsing a validation split message."""
        msg = self._make_message("split_val__house_00000100__repeats_05")
        result = parse_queue_message(msg, "val")

        assert result["split"] == "val"
        assert result["house_index"] == 100
        assert result["repeats"] == 5

    def test_parse_test_split(self):
        """Test parsing a test split message."""
        msg = self._make_message("split_test__house_00012345__repeats_10")
        result = parse_queue_message(msg, "test")

        assert result["split"] == "test"
        assert result["house_index"] == 12345
        assert result["repeats"] == 10

    def test_parse_train_unseen_split(self):
        """Test parsing a train_unseen split message."""
        msg = self._make_message("split_train_unseen__house_00000050__repeats_03")
        result = parse_queue_message(msg, "train_unseen")

        assert result["split"] == "train_unseen"
        assert result["house_index"] == 50
        assert result["repeats"] == 3

    def test_mismatched_split_raises(self):
        """Test that mismatched split raises assertion error."""
        msg = self._make_message("split_train__house_00000001__repeats_02")
        with pytest.raises(AssertionError):
            parse_queue_message(msg, "val")

    def test_invalid_characters_raises(self):
        """Test that invalid characters in ID raise assertion error."""
        msg = self._make_message("split_train__house_00000001__repeats_02!")
        with pytest.raises(AssertionError):
            parse_queue_message(msg, "train")


class TestSkipKeys:
    """Test skip_keys function."""

    def test_skip_single_key(self):
        """Test skipping a single key."""
        obs = {"key1": "value1", "key2": "value2", "key3": "value3"}
        result = skip_keys(obs, ["key2"])

        assert result == {"key1": "value1", "key3": "value3"}
        assert "key2" not in result

    def test_skip_multiple_keys(self):
        """Test skipping multiple keys."""
        obs = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = skip_keys(obs, ["a", "c"])

        assert result == {"b": 2, "d": 4}

    def test_skip_nonexistent_key(self):
        """Test skipping keys that don't exist."""
        obs = {"key1": "value1"}
        result = skip_keys(obs, ["nonexistent"])

        assert result == {"key1": "value1"}

    def test_skip_no_keys(self):
        """Test with empty skip list."""
        obs = {"key1": "value1", "key2": "value2"}
        result = skip_keys(obs, [])

        assert result == obs

    def test_skip_all_keys(self):
        """Test skipping all keys."""
        obs = {"key1": "value1", "key2": "value2"}
        result = skip_keys(obs, ["key1", "key2"])

        assert result == {}

    def test_empty_dict(self):
        """Test with empty input dictionary."""
        obs = {}
        result = skip_keys(obs, ["key1"])

        assert result == {}

    def test_preserves_value_types(self):
        """Test that value types are preserved."""
        obs = {
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "skip": "skip_me",
        }
        result = skip_keys(obs, ["skip"])

        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["list"] == [1, 2, 3]
        assert result["dict"] == {"nested": "value"}
