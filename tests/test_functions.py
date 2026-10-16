import numpy as np
import pytest

from ncp_homotopy import functions as fn

GRID = np.linspace(-2.0, 2.0, 41)
A, B = np.meshgrid(GRID, GRID)
A, B = A.ravel(), B.ravel()
IN_SOLUTION_SET = (A >= 0) & (B >= 0) & (A * B == 0)


@pytest.mark.parametrize("phi", [fn.phi_min, fn.phi_fb])
def test_ncp_function_zero_set(phi):
    values = phi(A, B)
    assert np.all(np.abs(values[IN_SOLUTION_SET]) < 1e-14)
    assert np.all(np.abs(values[~IN_SOLUTION_SET]) > 1e-14)


def test_chks_reduces_to_min_at_zero_mu():
    assert np.allclose(fn.phi_chks(A, B, 0.0), fn.phi_min(A, B))


@pytest.mark.parametrize("mu", [1e-6, 1e-3, 0.1, 1.0, 5.0])
def test_smoothing_error_bounds(mu):
    assert np.all(np.abs(fn.phi_chks(A, B, mu) - fn.phi_min(A, B)) <= 2 * mu + 1e-14)
    assert np.all(
        np.abs(fn.phi_fb_smooth(A, B, mu) - fn.phi_fb(A, B)) <= np.sqrt(2) * mu + 1e-14
    )


def _central_difference(g, *args, index, h=1e-6):
    plus = list(args)
    minus = list(args)
    plus[index] = plus[index] + h
    minus[index] = minus[index] - h
    return (g(*plus) - g(*minus)) / (2 * h)


def test_chks_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    mu = 0.3
    analytic = fn.phi_chks_grad(a, b, mu)
    for index in range(3):
        numeric = _central_difference(fn.phi_chks, a, b, mu, index=index)
        assert np.allclose(analytic[index], numeric, atol=1e-7)


def test_fb_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    for analytic, g, extra in [
        (fn.phi_fb_grad(a, b), fn.phi_fb, ()),
        (fn.phi_fb_smooth_grad(a, b, 0.2), fn.phi_fb_smooth, (0.2,)),
    ]:
        for index in range(2):
            numeric = _central_difference(g, a, b, *extra, index=index)
            assert np.allclose(analytic[index], numeric, atol=1e-7)


def test_generalized_gradient_at_kinks_is_finite():
    da, db = fn.phi_fb_grad(0.0, 0.0)
    assert np.isfinite(da) and np.isfinite(db)
    assert np.all(np.isfinite(fn.phi_chks_grad(1.0, 1.0, 0.0)))


def test_residual_measures():
    x = np.array([0.0, 2.0, 0.0])
    fx = np.array([3.0, 0.0, 0.0])
    assert fn.natural_residual(x, fx) == 0.0
    assert fn.complementarity_gap(x, fx) == 0.0
    assert fn.is_ncp_solution(x, fx)
    assert not fn.is_ncp_solution(np.array([1.0]), np.array([1.0]))
    assert not fn.is_ncp_solution(np.array([-1.0]), np.array([0.0]))
