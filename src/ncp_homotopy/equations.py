"""Homotopy solvers for square nonlinear systems F(x) = 0.

Two classical homotopies are provided, both with the trivial zero x = a
at lam = 0:

* fixed point:  H(x, lam) = lam F(x) + (1 - lam)(x - a)
* Newton:       H(x, lam) = F(x) - (1 - lam) F(a)
"""

import numpy as np

from .tracking import track


def fixed_point_homotopy(F, JF, a):
    """Return (H, J) for the fixed-point homotopy started at ``a``."""
    a = np.asarray(a, dtype=float)
    eye = np.eye(a.size)

    def H(y):
        x, lam = y[:-1], y[-1]
        return lam * F(x) + (1.0 - lam) * (x - a)

    def J(y):
        x, lam = y[:-1], y[-1]
        dx = lam * JF(x) + (1.0 - lam) * eye
        dlam = F(x) - (x - a)
        return np.column_stack([dx, dlam])

    return H, J


def newton_homotopy(F, JF, a):
    """Return (H, J) for the Newton homotopy started at ``a``."""
    a = np.asarray(a, dtype=float)
    Fa = np.asarray(F(a), dtype=float)

    def H(y):
        x, lam = y[:-1], y[-1]
        return F(x) - (1.0 - lam) * Fa

    def J(y):
        return np.column_stack([JF(y[:-1]), Fa])

    return H, J


_HOMOTOPIES = {"fixed_point": fixed_point_homotopy, "newton": newton_homotopy}


def solve(F, JF, a, method="fixed_point", **track_options):
    """Solve F(x) = 0 by tracking a homotopy from the starting point ``a``.

    ``F`` and ``JF`` take and return numpy arrays (use shape (1,) for a
    scalar equation). Extra keyword arguments go to :func:`tracking.track`.
    """
    try:
        build = _HOMOTOPIES[method]
    except KeyError:
        raise ValueError(f"unknown homotopy {method!r}") from None
    H, J = build(F, JF, a)
    return track(H, J, a, **track_options)
