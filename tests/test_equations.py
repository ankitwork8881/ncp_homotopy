import numpy as np
import pytest

from ncp_homotopy import equations, problems
from ncp_homotopy.tracking import unit_tangent


def test_unit_tangent_spans_null_space_and_keeps_orientation():
    rng = np.random.default_rng(0)
    jac = rng.standard_normal((4, 5))
    t = unit_tangent(jac)
    assert np.isclose(np.linalg.norm(t), 1.0)
    assert np.allclose(jac @ t, 0.0, atol=1e-12)
    assert t[-1] > 0
    assert np.dot(unit_tangent(jac, previous=-t), -t) > 0


@pytest.mark.parametrize("method", ["fixed_point", "newton"])
def test_linear_system(method):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    b = rng.standard_normal(4)
    result = equations.solve(lambda x: A @ x - b, lambda x: A, np.zeros(4), method=method)
    assert result.success
    assert np.allclose(result.u, np.linalg.solve(A, b), atol=1e-10)


@pytest.mark.parametrize("method", ["fixed_point", "newton"])
@pytest.mark.parametrize(
    "problem", [problems.freudenstein_roth(), problems.cubic_with_coupling(6)]
)
def test_nonlinear_systems(problem, method):
    result = equations.solve(problem.f, problem.jac, problem.start, method=method)
    assert result.success
    assert np.linalg.norm(problem.f(result.u)) < 1e-10
    for known in problem.solutions:
        assert np.allclose(result.u, known, atol=1e-8)


def test_fixed_point_homotopy_escapes_narrow_valley():
    p = problems.narrow_valley()
    result = equations.solve(p.f, p.jac, p.start, method="fixed_point")
    assert result.success
    assert abs(result.u[0]) < 1e-10


def test_damped_newton_gets_trapped_in_narrow_valley():
    # Baseline showing why the homotopy matters on this problem: Newton's
    # method with backtracking on f^2/2 stalls away from the root.
    p = problems.narrow_valley()
    x = p.start.copy()
    for _ in range(500):
        r = p.f(x)
        step = -r / p.jac(x)[0]
        t = 1.0
        while 0.5 * p.f(x + t * step)[0] ** 2 > 0.5 * r[0] ** 2 and t > 1e-12:
            t *= 0.5
        x = x + t * step
    assert abs(x[0]) > 0.1


def test_path_starts_at_start_point_and_ends_at_lambda_one():
    p = problems.cubic_with_coupling(3)
    result = equations.solve(p.f, p.jac, p.start)
    assert np.allclose(result.path[0], np.append(p.start, 0.0))
    assert result.path[-1][-1] == 1.0
    assert result.arc_length > 0


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        equations.solve(lambda x: x, lambda x: np.eye(1), np.zeros(1), method="bogus")
