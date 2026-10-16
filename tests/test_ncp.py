import itertools

import numpy as np
import pytest

from ncp_homotopy import functions as fn
from ncp_homotopy import ncp, problems

METHODS = list(ncp.METHODS)


def brute_force_lcp(M, q):
    """Solve a small LCP exactly by enumerating the support of x."""
    n = q.size
    for support in itertools.product([False, True], repeat=n):
        idx = np.flatnonzero(support)
        x = np.zeros(n)
        if idx.size:
            try:
                x[idx] = np.linalg.solve(M[np.ix_(idx, idx)], -q[idx])
            except np.linalg.LinAlgError:
                continue
        if fn.is_ncp_solution(x, M @ x + q, tol=1e-10):
            return x
    raise AssertionError("no LCP solution found")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(5))
def test_pd_lcp_matches_enumeration(method, seed):
    p = problems.random_pd_lcp(5, seed=seed)
    M, q = p.jac(None), p.f(np.zeros(5))
    result = ncp.solve(p.f, p.jac, p.start, method=method)
    assert result.success
    assert np.allclose(result.x, brute_force_lcp(M, q), atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_monotone_nonlinear_ncp(method):
    p = problems.monotone_cubic_ncp()
    result = ncp.solve(p.f, p.jac, p.start, method=method)
    assert result.success
    assert fn.is_ncp_solution(result.x, result.fx)
    assert result.gap < 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_kojima_shindo_reaches_a_known_solution(method):
    p = problems.kojima_shindo()
    result = ncp.solve(p.f, p.jac, p.start, method=method)
    assert result.success
    assert any(np.allclose(result.x, s, atol=1e-7) for s in p.solutions)


def test_methods_agree_on_unique_solution():
    p = problems.monotone_cubic_ncp(n=8, seed=7)
    solutions = [ncp.solve(p.f, p.jac, p.start, method=m).x for m in METHODS]
    for x in solutions[1:]:
        assert np.allclose(x, solutions[0], atol=1e-8)


def test_homotopy_path_alone_is_accurate():
    # Without the semismooth Newton polish, the tracked end point already
    # solves the exact (mu = 0) reformulation.
    p = problems.random_pd_lcp(4, seed=11)
    result = ncp.homotopy(p.f, p.jac, p.start, polish=False, tol=1e-8)
    assert result.success


def test_smoothed_homotopy_jacobian_matches_finite_differences():
    p = problems.kojima_shindo()
    a = np.concatenate([p.start, p.f(p.start)])
    H, J = ncp.smoothed_homotopy(p.f, p.jac, a, beta=0.7)
    w = np.concatenate([np.array([0.4, 0.9, 1.3, 0.2, 0.5, -0.3, 1.1, 0.8]), [0.6]])
    h = 1e-6
    numeric = np.column_stack(
        [(H(w + h * e) - H(w - h * e)) / (2 * h) for e in np.eye(w.size)]
    )
    assert np.allclose(J(w), numeric, atol=1e-6)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        ncp.solve(lambda x: x, lambda x: np.eye(1), np.ones(1), method="bogus")
