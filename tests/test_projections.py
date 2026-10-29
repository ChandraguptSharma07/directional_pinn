"""Tests for the residual decomposition (src/analysis/projections.py)."""

import numpy as np
import pytest

from src.analysis.projections import decompose_residual


@pytest.fixture
def batch():
    """200 random residuals in 3-D with sizes spread over 7 decades, and random unit directions."""
    rng = np.random.default_rng(12345)
    r = rng.normal(size=(200, 3)) * 10.0 ** rng.uniform(-6.0, 1.0, size=(200, 1))
    u = rng.normal(size=(200, 3))
    return r, u / np.linalg.norm(u, axis=1, keepdims=True)


def test_parts_add_up_to_residual(batch):
    r, u = batch
    parts = decompose_residual(r, u)
    mismatch = np.linalg.norm(parts.r_u + parts.r_perp - r, axis=1)
    assert np.all(mismatch <= 1e-14 * np.linalg.norm(r, axis=1))


def test_perpendicular_part_is_orthogonal_to_direction(batch):
    r, u = batch
    parts = decompose_residual(r, u)
    dots = np.sum(parts.r_perp * u, axis=1)
    assert np.all(np.abs(dots) <= 1e-14 * np.linalg.norm(r, axis=1))


def test_pythagoras(batch):
    r, u = batch
    parts = decompose_residual(r, u)
    np.testing.assert_allclose(parts.R**2, parts.R_u**2 + parts.R_perp**2, rtol=1e-12)


def test_three_four_five_example():
    parts = decompose_residual(np.array([3.0, 4.0]), np.array([1.0, 0.0]))
    assert parts.R == pytest.approx(5.0)
    assert parts.R_u == pytest.approx(3.0)
    assert parts.R_perp == pytest.approx(4.0)
    assert parts.alignment == pytest.approx(0.6)
    np.testing.assert_allclose(parts.r_u, [3.0, 0.0])
    np.testing.assert_allclose(parts.r_perp, [0.0, 4.0])


def test_alignment_limits(batch):
    u = np.array([1.0, 0.0, 0.0])
    parallel = decompose_residual(np.array([[3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]), u)
    np.testing.assert_allclose(parallel.alignment, 1.0, rtol=1e-12)
    perpendicular = decompose_residual(np.array([[0.0, 3.0, 0.0], [0.0, 0.0, -2.0]]), u)
    np.testing.assert_array_equal(perpendicular.alignment, 0.0)
    r, u = batch
    alignment = decompose_residual(r, u).alignment
    assert np.all((alignment >= 0.0) & (alignment <= 1.0))


def test_sign_of_direction_does_not_matter(batch):
    r, u = batch
    plus, minus = decompose_residual(r, u), decompose_residual(r, -u)
    for name in ("r_u", "r_perp", "R", "R_u", "R_perp", "alignment"):
        np.testing.assert_array_equal(getattr(plus, name), getattr(minus, name))


def test_single_direction_is_used_for_every_residual(batch):
    r, _ = batch
    u = np.array([0.6, 0.0, 0.8])
    single = decompose_residual(r, u)
    repeated = decompose_residual(r, np.tile(u, (len(r), 1)))
    np.testing.assert_array_equal(single.R_u, repeated.R_u)
    np.testing.assert_array_equal(single.r_perp, repeated.r_perp)


def test_zero_residual_gives_zero_alignment():
    parts = decompose_residual(np.zeros((1, 3)), np.array([0.0, 0.0, 1.0]))
    assert parts.alignment[0] == 0.0


def test_rejects_direction_that_is_not_unit_length():
    with pytest.raises(ValueError):
        decompose_residual(np.ones(2), np.array([1.0, 1.0]))
