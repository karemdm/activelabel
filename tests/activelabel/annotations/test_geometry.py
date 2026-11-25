"""Tests of the geometry shared by the annotation readers."""

from __future__ import annotations

import pytest

from activelabel.annotations.geometry import enclosing_box


def test_no_turn_returns_the_box_itself() -> None:
    assert enclosing_box([10, 10, 30, 20], 0) == [10, 10, 30, 20]


def test_a_full_turn_is_no_turn() -> None:
    assert enclosing_box([10, 10, 30, 20], 360) == [10, 10, 30, 20]


def test_a_quarter_turn_swaps_width_and_height_about_the_centre() -> None:
    assert enclosing_box([10, 10, 30, 20], 90) == pytest.approx([15, 5, 25, 25])


def test_a_square_turned_45_degrees_grows_to_its_diagonal() -> None:
    half = 50**0.5
    assert enclosing_box([0, 0, 10, 10], 45) == pytest.approx([5 - half, 5 - half, 5 + half, 5 + half])


def test_negative_angles_are_the_same_as_their_positive_turn() -> None:
    assert enclosing_box([0, 0, 10, 20], -90) == pytest.approx(enclosing_box([0, 0, 10, 20], 270))
    assert enclosing_box([0, 0, 10, 10], 30) == pytest.approx(enclosing_box([0, 0, 10, 10], -30))
