"""Tests for QA generator pure functions in sims.qa.generators."""

import random

import numpy as np
import pytest

from sims.qa.generators import QA_GEN_FNS
from sims.qa.generators.common import (
    clean_obj_type,
    gen_count_options,
    gen_distance_options,
    gen_mc_question,
)
from sims.qa.generators.spatial import calc_3d_bbox_distance_between_objects


# ---------- QA_GEN_FNS registry ----------


class TestQAGenFNS:
    def test_has_expected_count(self):
        assert len(QA_GEN_FNS) == 33

    def test_no_duplicate_keys(self):
        # Implicitly tested by the RuntimeError guard in __init__.py,
        # but verify the assembled dict has distinct keys.
        keys = list(QA_GEN_FNS.keys())
        assert len(keys) == len(set(keys))

    def test_all_values_are_callable(self):
        for key, fn in QA_GEN_FNS.items():
            assert callable(fn), f"QA_GEN_FNS[{key!r}] is not callable"


# ---------- clean_obj_type ----------


class TestCleanObjType:
    def test_removes_obja_prefix(self):
        assert clean_obj_type("ObjaChair") == "chair"

    def test_splits_camel_case(self):
        assert clean_obj_type("DiningTable") == "dining table"

    def test_obja_prefix_and_camel_case(self):
        assert clean_obj_type("ObjaFloorLamp") == "floor lamp"

    def test_single_word(self):
        assert clean_obj_type("Chair") == "chair"

    def test_empty_string(self):
        assert clean_obj_type("") == ""


# ---------- gen_mc_question ----------


class TestGenMCQuestion:
    def test_basic_generation(self):
        mc_q, mc_a, mc_opts = gen_mc_question(
            "What color?", "red", ["red", "blue", "green"], shuffle_options=False
        )
        assert mc_a == "A"
        assert "A. red" in mc_opts[0]
        assert "B. blue" in mc_opts[1]
        assert "C. green" in mc_opts[2]
        assert mc_q.startswith("What color?")

    def test_answer_letter_matches(self):
        # With no shuffle, answer should always be at its original position
        _, mc_a, mc_opts = gen_mc_question(
            "Q?", "blue", ["red", "blue"], shuffle_options=False
        )
        assert mc_a == "B"

    def test_shuffled_answer_is_correct(self):
        random.seed(42)
        _, mc_a, mc_opts = gen_mc_question(
            "Q?", "yes", ["yes", "no"], shuffle_options=True
        )
        # The answer letter should point to the option containing "yes"
        answer_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(mc_a)
        assert "yes" in mc_opts[answer_idx]

    def test_options_not_mutated(self):
        original = ["a", "b", "c"]
        copy = original.copy()
        gen_mc_question("Q?", "a", original, shuffle_options=True)
        assert original == copy


# ---------- gen_count_options ----------


class TestGenCountOptions:
    def test_returns_four_options(self):
        options = gen_count_options(5)
        assert len(options) == 4

    def test_answer_included(self):
        for answer in [1, 3, 7, 10]:
            options = gen_count_options(answer)
            assert answer in options

    def test_all_positive(self):
        # Even for answer=1, offsets should not produce non-positive values
        random.seed(0)
        for _ in range(20):
            options = gen_count_options(1)
            assert all(opt >= 1 for opt in options)

    def test_no_duplicates(self):
        random.seed(42)
        options = gen_count_options(5)
        assert len(options) == len(set(options))


# ---------- gen_distance_options ----------


class TestGenDistanceOptions:
    def test_returns_correct_count(self):
        options = gen_distance_options(2.5, num_options=4)
        assert len(options) == 4

    def test_answer_included(self):
        options = gen_distance_options(3.0, num_options=4, digits=1)
        assert 3.0 in options

    def test_zero_answer(self):
        options = gen_distance_options(0)
        assert len(options) == len(set(options)) == 4
        assert 0 in options

    def test_negative_answer_raises(self):
        with pytest.raises(ValueError, match="Invalid negative answer"):
            gen_distance_options(-1.0)

    def test_distinct_values(self):
        random.seed(42)
        options = gen_distance_options(5.0)
        assert len(options) == len(set(options))


# ---------- calc_3d_bbox_distance_between_objects ----------


class TestCalcBboxDistance:
    def test_separated_boxes(self):
        bbox1 = {"min": [0, 0, 0], "max": [1, 1, 1]}
        bbox2 = {"min": [3, 0, 0], "max": [4, 1, 1]}
        dist = calc_3d_bbox_distance_between_objects(bbox1, bbox2)
        assert dist == pytest.approx(2.0)

    def test_touching_boxes(self):
        bbox1 = {"min": [0, 0, 0], "max": [1, 1, 1]}
        bbox2 = {"min": [1, 0, 0], "max": [2, 1, 1]}
        dist = calc_3d_bbox_distance_between_objects(bbox1, bbox2)
        assert dist == pytest.approx(0.0)

    def test_overlapping_boxes(self):
        bbox1 = {"min": [0, 0, 0], "max": [2, 2, 2]}
        bbox2 = {"min": [1, 1, 1], "max": [3, 3, 3]}
        dist = calc_3d_bbox_distance_between_objects(bbox1, bbox2)
        assert dist == pytest.approx(0.0)

    def test_diagonal_separation(self):
        bbox1 = {"min": [0, 0, 0], "max": [1, 1, 1]}
        bbox2 = {"min": [2, 2, 2], "max": [3, 3, 3]}
        dist = calc_3d_bbox_distance_between_objects(bbox1, bbox2)
        expected = np.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)
        assert dist == pytest.approx(expected)

    def test_same_box(self):
        bbox = {"min": [1, 2, 3], "max": [4, 5, 6]}
        dist = calc_3d_bbox_distance_between_objects(bbox, bbox)
        assert dist == pytest.approx(0.0)

    def test_separation_in_one_axis(self):
        bbox1 = {"min": [0, 0, 0], "max": [1, 1, 1]}
        bbox2 = {"min": [0, 0, 4], "max": [1, 1, 5]}
        dist = calc_3d_bbox_distance_between_objects(bbox1, bbox2)
        assert dist == pytest.approx(3.0)
