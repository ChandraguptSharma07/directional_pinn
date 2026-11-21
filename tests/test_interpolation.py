"""Checks that coarse reconstruction and trailing integration use the intended signal."""

import numpy as np
import pytest

from src.analysis.interpolation import reconstruct_residual, trailing_mean_magnitude


def test_linear_reconstruction_is_exact_for_affine_vector_residual():
    t = np.arange(0.0, 1.01, 0.01)
    residual = np.column_stack([1.0 + 2.0 * t, -3.0 * t])
    np.testing.assert_allclose(reconstruct_residual(t, residual, 7, "linear"), residual, atol=1e-14)
    held = reconstruct_residual(t, residual, 7, "hold")
    np.testing.assert_allclose(held[8], residual[7])
    np.testing.assert_allclose(held[-1], residual[-1])


def test_trailing_average_of_positive_affine_scalar_is_exact():
    t = np.arange(0.0, 1.01, 0.01)
    residual = (2.0 + 3.0 * t)[:, None]
    average = trailing_mean_magnitude(t, residual, 0.2)
    assert np.isnan(average[:20]).all()
    np.testing.assert_allclose(average[20:], 2.0 + 3.0 * (t[20:] - 0.1), atol=1e-13)


def test_invalid_sampling_or_width_is_rejected():
    t = np.arange(0.0, 0.11, 0.01)
    residual = np.ones((len(t), 2))
    with pytest.raises(ValueError):
        reconstruct_residual(t, residual, 0, "linear")
    with pytest.raises(ValueError):
        reconstruct_residual(t, residual, 2, "cubic")
    with pytest.raises(ValueError):
        trailing_mean_magnitude(t, residual, 0.015)
