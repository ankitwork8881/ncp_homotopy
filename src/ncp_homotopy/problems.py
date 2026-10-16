"""Test problems for the equation and NCP solvers.

Each problem is a small namespace with the map, its Jacobian, a suggested
starting point and, where known, the solution(s).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    name: str
    f: object
    jac: object
    start: np.ndarray
    solutions: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Nonlinear equations
# --------------------------------------------------------------------------

def narrow_valley():
    """1-D equation whose merit function f^2/2 has its minimum in a narrow
    valley at x = 0, surrounded by spurious local minima that trap
    descent-based methods."""

    def f(x):
        t = x[0]
        return np.array(
            [np.arctan(100 * t) / np.pi + np.sin(5 * t / (t ** 2 + 0.2)) / 2 + 0.1 * t]
        )

    def jac(x):
        t = x[0]
        q = t ** 2 + 0.2
        inner = 5 * t / q
        d_inner = 5 * (0.2 - t ** 2) / q ** 2
        return np.array(
            [[100 / (np.pi * (1 + (100 * t) ** 2)) + np.cos(inner) * d_inner / 2 + 0.1]]
        )

    return Problem("narrow_valley", f, jac, np.array([1.0]), [np.array([0.0])])


def freudenstein_roth():
    """Freudenstein-Roth function. Plain Newton from the standard start
    (0.5, -2) stalls near a local minimiser of the residual."""

    def f(x):
        x1, x2 = x
        return np.array(
            [
                -13 + x1 + ((5 - x2) * x2 - 2) * x2,
                -29 + x1 + ((x2 + 1) * x2 - 14) * x2,
            ]
        )

    def jac(x):
        _, x2 = x
        return np.array(
            [
                [1.0, 10 * x2 - 3 * x2 ** 2 - 2],
                [1.0, 3 * x2 ** 2 + 2 * x2 - 14],
            ]
        )

    return Problem(
        "freudenstein_roth", f, jac, np.array([0.5, -2.0]), [np.array([5.0, 4.0])]
    )


def cubic_with_coupling(n=5):
    """F(x) = x^3 + T x - 1, where T is the tridiagonal second-difference
    matrix; strictly monotone, so the solution is unique."""
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

    def f(x):
        return x ** 3 + T @ x - 1.0

    def jac(x):
        return np.diag(3 * x ** 2) + T

    return Problem(f"cubic_with_coupling_{n}", f, jac, np.zeros(n))


# --------------------------------------------------------------------------
# Complementarity problems
# --------------------------------------------------------------------------

def lcp(M, q, start=None):
    """Linear complementarity problem: f(x) = M x + q."""
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)
    start = np.ones(q.size) if start is None else np.asarray(start, dtype=float)
    return Problem("lcp", lambda x: M @ x + q, lambda x: M, start)


def random_pd_lcp(n, seed=0):
    """LCP with a random positive definite (non-symmetric) matrix, so the
    solution exists and is unique."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    S = rng.standard_normal((n, n))
    M = B @ B.T + n * np.eye(n) + (S - S.T)
    q = rng.standard_normal(n) * 3
    return lcp(M, q)


def kojima_shindo():
    """Kojima-Shindo NCP in R^4: a nonmonotone problem with a degenerate
    solution (sqrt(6)/2, 0, 0, 1/2) and a nondegenerate one (1, 0, 3, 0)."""

    def f(x):
        x1, x2, x3, x4 = x
        return np.array(
            [
                3 * x1 ** 2 + 2 * x1 * x2 + 2 * x2 ** 2 + x3 + 3 * x4 - 6,
                2 * x1 ** 2 + x1 + x2 ** 2 + 3 * x3 + 2 * x4 - 2,
                3 * x1 ** 2 + x1 * x2 + 2 * x2 ** 2 + 2 * x3 + 3 * x4 - 1,
                x1 ** 2 + 3 * x2 ** 2 + 2 * x3 + 3 * x4 - 3,
            ]
        )

    def jac(x):
        x1, x2, _, _ = x
        return np.array(
            [
                [6 * x1 + 2 * x2, 2 * x1 + 4 * x2, 1, 3],
                [4 * x1 + 1, 2 * x2, 3, 2],
                [6 * x1 + x2, x1 + 4 * x2, 2, 3],
                [2 * x1, 6 * x2, 2, 3],
            ],
            dtype=float,
        )

    return Problem(
        "kojima_shindo",
        f,
        jac,
        np.ones(4),
        [np.array([np.sqrt(6) / 2, 0, 0, 0.5]), np.array([1.0, 0, 3, 0])],
    )


def monotone_cubic_ncp(n=6, seed=1):
    """f(x) = x^3 + M x + q with M positive semidefinite: a monotone,
    nonlinear NCP."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    M = B @ B.T
    q = rng.standard_normal(n) * 2

    def f(x):
        return x ** 3 + M @ x + q

    def jac(x):
        return np.diag(3 * x ** 2) + M

    return Problem(f"monotone_cubic_{n}", f, jac, np.ones(n))
