# tests/test_string_utils.py
"""Tests for sims.utils.string_utils module."""

import pytest
import numpy as np

from sims.utils.string_utils import (
    convert_string_to_byte,
    convert_byte_to_string,
    json_templated_spec_to_dict,
    strings_exist_in_dict_or_list,
)


class TestStringByteConversion:
    """Test convert_string_to_byte and convert_byte_to_string roundtrip."""

    @pytest.mark.parametrize(
        "test_string,max_len",
        [
            ("hello", 10),
            ("hello", 5),
            ("", 10),
            ("a", 1),
            ("test string with spaces", 50),
            ("special!@#$%", 20),
            ("unicode_test", 20),
        ],
    )
    def test_roundtrip_conversion(self, test_string, max_len):
        """Test that encoding and decoding produces the original string."""
        encoded = convert_string_to_byte(test_string, max_len)
        decoded = convert_byte_to_string(encoded, max_len)
        assert decoded == test_string

    def test_convert_string_to_byte_returns_numpy_array(self):
        """Test that the function returns a numpy array."""
        result = convert_string_to_byte("test", 10)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_convert_byte_to_string_infers_max_len(self):
        """Test that max_len can be inferred from array shape."""
        encoded = convert_string_to_byte("test", 10)
        decoded = convert_byte_to_string(encoded)
        assert decoded == "test"


class TestJsonTemplatedSpecToDict:
    """Test json_templated_spec_to_dict function."""

    def test_simple_json_parsing(self):
        """Test parsing simple JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = json_templated_spec_to_dict(json_str)
        assert result == {"key": "value", "number": 42}

    def test_nested_json_parsing(self):
        """Test parsing nested JSON string."""
        json_str = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = json_templated_spec_to_dict(json_str)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_empty_object(self):
        """Test parsing empty JSON object."""
        result = json_templated_spec_to_dict("{}")
        assert result == {}


class TestStringsExistInDictOrList:
    """Test strings_exist_in_dict_or_list function."""

    def test_string_in_simple_dict(self):
        """Test finding string in simple dictionary value."""
        data = {"key": "hello world"}
        assert strings_exist_in_dict_or_list(data, ["hello"]) is True
        assert strings_exist_in_dict_or_list(data, ["goodbye"]) is False

    def test_string_in_nested_dict(self):
        """Test finding string in nested dictionary."""
        data = {"outer": {"inner": "target string"}}
        assert strings_exist_in_dict_or_list(data, ["target"]) is True
        assert strings_exist_in_dict_or_list(data, ["missing"]) is False

    def test_string_in_list(self):
        """Test finding string in list."""
        data = ["first", "second", "third"]
        assert strings_exist_in_dict_or_list(data, ["second"]) is True
        assert strings_exist_in_dict_or_list(data, ["fourth"]) is False

    def test_string_in_nested_list(self):
        """Test finding string in nested list."""
        data = [["nested", "list"], "outer"]
        assert strings_exist_in_dict_or_list(data, ["nested"]) is True

    def test_string_in_mixed_structure(self):
        """Test finding string in mixed dict/list structure."""
        data = {
            "items": [
                {"name": "item1", "value": "target_value"},
                {"name": "item2", "value": "other"},
            ]
        }
        assert strings_exist_in_dict_or_list(data, ["target_value"]) is True
        assert strings_exist_in_dict_or_list(data, ["item1"]) is True
        assert strings_exist_in_dict_or_list(data, ["missing"]) is False

    def test_multiple_target_strings(self):
        """Test searching for multiple target strings."""
        data = {"key": "hello world"}
        assert strings_exist_in_dict_or_list(data, ["hello", "goodbye"]) is True
        assert strings_exist_in_dict_or_list(data, ["foo", "bar"]) is False

    def test_none_target_strings(self):
        """Test with None as target_strings."""
        data = {"key": "value"}
        assert strings_exist_in_dict_or_list(data, None) is False

    def test_empty_data_structures(self):
        """Test with empty data structures."""
        assert strings_exist_in_dict_or_list({}, ["target"]) is False
        assert strings_exist_in_dict_or_list([], ["target"]) is False

    def test_non_string_values(self):
        """Test with non-string values in data."""
        data = {"number": 42, "boolean": True, "none": None}
        assert strings_exist_in_dict_or_list(data, ["42"]) is False

    def test_partial_string_match(self):
        """Test that partial string matches work."""
        data = {"key": "hello world"}
        assert strings_exist_in_dict_or_list(data, ["llo"]) is True
        assert strings_exist_in_dict_or_list(data, ["wor"]) is True
