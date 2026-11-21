"""Tests for the future-error decomposition and the statistics of the central experiment."""

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.future_error import created_error, future_error, horizon_steps, linearized_error
from src.analysis.samples import join_free_indices
from src.analysis.metrics import (
    alignment_effect,
    bootstrap_delta_r_squared,
    bootstrap_mean,
    cluster_sums,
    leave_one_group_out,
    ols_fit,
    ols_predict,
    partial_spearman,
    pooled_r_squared,
    r_squared,
)
from src.dynamics.tangent_dynamics import propagators_over
from src.systems.linear_saddle import saddle, saddle_forced_error


def test_horizon_steps():
    assert horizon_steps(0.05, 0.002) == 25
    with pytest.raises(ValueError):
        horizon_steps(0.003, 0.002)


def test_piecewise_join_times_are_excluded_from_predictor_samples():
    t = np.arange(0.0, 1.0, 0.01)
    indices = np.arange(len(t))
    kept = join_free_indices(t, indices, 0.25)
    assert 0 in kept
    assert all(i not in kept for i in (25, 50, 75))
    assert len(kept) == len(indices) - 3
    np.testing.assert_array_equal(join_free_indices(t, indices, None), indices)
    with pytest.raises(ValueError):
        join_free_indices(t, indices, -0.25)


def test_future_and_created_error_on_the_linear_saddle():
    """Constant residual r on the saddle: e(t) = Psi(t) r exactly, so the error created in
    [t, t + Delta] is Psi(Delta) r whatever t is (Duhamel)."""
    h, m = 0.01, 20
    t = np.arange(0.0, 2.0 + h / 2, h)
    r = np.array([1e-3, 2e-3])
    error = saddle_forced_error(t, r)
    props = np.broadcast_to(expm(saddle().A * h), (len(t) - 1, 2, 2))
    np.testing.assert_allclose(future_error(error, m), np.linalg.norm(error[m:], axis=1))
    created = created_error(error, propagators_over(props, m), m)
    expected = np.linalg.norm(saddle().forced_error_operator(m * h) @ r)
    np.testing.assert_allclose(created, expected, rtol=1e-10)


def test_linearized_error_reproduces_the_exact_linear_error():
    h = 0.001
    t = np.arange(0.0, 2.0 + h / 2, h)
    r = np.array([1e-3, -2e-3])
    e0 = np.array([1e-4, 3e-4])
    exact = saddle_forced_error(t, r) + np.column_stack([e0[0] * np.exp(t), e0[1] * np.exp(-t)])
    props = np.broadcast_to(expm(saddle().A * h), (len(t) - 1, 2, 2))
    reconstructed = linearized_error(e0, np.tile(r, (len(t), 1)), props, h)
    np.testing.assert_allclose(reconstructed, exact, rtol=1e-6, atol=1e-10)  # trapezoidal error is O(h^2)


def test_partial_spearman_removes_a_shared_driver():
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    x, y = z + 0.1 * rng.normal(size=2000), z + 0.1 * rng.normal(size=2000)
    assert np.corrcoef(x, y)[0, 1] > 0.9
    assert abs(partial_spearman(x, y, z)) < 0.1


def test_alignment_effect_detects_and_rejects_an_effect():
    rng = np.random.default_rng(1)
    magnitude = rng.lognormal(size=3000)
    alignment = rng.uniform(size=3000)
    noise = 0.1 * rng.normal(size=3000)
    with_effect = np.log(magnitude) + alignment + noise
    without = np.log(magnitude) + noise
    assert alignment_effect(magnitude, alignment, with_effect) == pytest.approx(2 / 3, abs=0.1)
    assert abs(alignment_effect(magnitude, alignment, without)) < 0.1  # sampling noise within magnitude bins


def test_ols_and_leave_one_group_out():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 2))
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.01 * rng.normal(size=600)
    np.testing.assert_allclose(ols_fit(X, y), [1.0, 2.0, -0.5], atol=0.01)
    groups = np.repeat(np.arange(6), 100)
    prediction = leave_one_group_out(X, y, groups)
    assert r_squared(y, prediction) > 0.999
    np.testing.assert_allclose(ols_predict(ols_fit(X, y), X), y, atol=0.05)


def test_pooled_r_squared_from_cluster_sums_matches_direct_computation():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    p0, p1 = y + rng.normal(size=500), y + 0.5 * rng.normal(size=500)
    clusters = rng.integers(0, 10, size=500)
    _, sums = cluster_sums(y, [p0, p1], clusters)
    np.testing.assert_allclose(pooled_r_squared(sums, np.ones(len(sums))), [r_squared(y, p0), r_squared(y, p1)])


def test_bootstrap_delta_r_squared_and_mean():
    rng = np.random.default_rng(4)
    y = rng.normal(size=1000)
    clusters = np.repeat(np.arange(20), 50)
    result = bootstrap_delta_r_squared(y, y + rng.normal(size=1000), y + 0.3 * rng.normal(size=1000), clusters,
                                       n_boot=300)
    assert result["delta"] > 0 and result["ci_low"] > 0
    null = bootstrap_delta_r_squared(y, y + rng.normal(size=1000), y + rng.normal(size=1000), clusters, n_boot=300)
    assert null["ci_low"] < 0 < null["ci_high"] or abs(null["delta"]) < 0.05
    summary = bootstrap_mean(np.array([1.0, 2.0, 3.0, np.nan]), n_boot=200)
    assert summary["mean"] == pytest.approx(2.0) and summary["n"] == 3
