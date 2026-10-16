"""Predictor-corrector tracking of a homotopy zero curve.

A homotopy H: R^n x R -> R^n is written as a function of y = (u, lam),
with lam in [0, 1]. Starting from a known zero at lam = 0, the zero curve
{y : H(y) = 0} is followed in arc length until it crosses lam = 1, where
u solves the target system.

Each step:

* takes the unit tangent as the null vector of the n x (n+1) Jacobian,
  oriented to keep the direction of travel (so turning points in lam are
  passed without special handling);
* predicts with an Euler step along the tangent;
* corrects back to the curve with minimum-norm (normal flow) Newton
  steps, halving the step length when the corrector fails.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackResult:
    """Outcome of tracking a zero curve."""

    u: np.ndarray
    lam: float
    success: bool
    message: str
    steps: int
    arc_length: float
    path: np.ndarray = field(repr=False)


def unit_tangent(jac, previous=None):
    """Unit null vector of an n x (n+1) Jacobian.

    Oriented to make an acute angle with ``previous``, or, on the first
    step, to point towards increasing lam.
    """
    q, _ = np.linalg.qr(jac.T, mode="complete")
    t = q[:, -1]
    if previous is None:
        sign = np.sign(t[-1])
    else:
        sign = np.sign(np.dot(t, previous))
    return t if sign >= 0 else -t


def _normal_flow_correct(H, J, w, tol, max_iter):
    for k in range(1, max_iter + 1):
        step = np.linalg.lstsq(J(w), -H(w), rcond=None)[0]
        w = w + step
        if not np.all(np.isfinite(w)):
            return False, w, k
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(w)):
            return True, w, k
    return False, w, max_iter


def _newton_at_end(H, J, guess, tol, max_iter):
    """Newton's method on u -> H(u, 1) starting from ``guess``."""
    y = guess.copy()
    y[-1] = 1.0
    n = y.size - 1
    for _ in range(max_iter):
        r = H(y)
        if np.linalg.norm(r) <= tol:
            break
        step = np.linalg.lstsq(J(y)[:, :n], -r, rcond=None)[0]
        y[:n] += step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(y[:n])):
            break
    return y


def track(
    H,
    J,
    u0,
    *,
    h0=0.05,
    h_min=1e-10,
    h_max=1.0,
    corrector_tol=1e-9,
    max_corrector=6,
    end_tol=1e-12,
    max_steps=20000,
    max_arc=1e4,
):
    """Follow the zero curve of ``H`` from (u0, 0) to lam = 1.

    Parameters
    ----------
    H : callable
        H(y) -> array of shape (n,), with y = (u, lam) of shape (n + 1,).
    J : callable
        J(y) -> array of shape (n, n + 1), the Jacobian [dH/du, dH/dlam].
    u0 : array_like
        Zero of H at lam = 0.
    """
    y = np.append(np.asarray(u0, dtype=float), 0.0)
    path = [y.copy()]
    tangent = None
    h = h0
    arc = 0.0

    for step in range(1, max_steps + 1):
        tangent = unit_tangent(J(y), tangent)

        while True:
            ok, w, iters = _normal_flow_correct(
                H, J, y + h * tangent, corrector_tol, max_corrector
            )
            # Reject steps that converge but jump backwards along the curve.
            if ok and np.dot(w - y, tangent) > 0:
                break
            h *= 0.5
            if h < h_min:
                return TrackResult(
                    y[:-1], y[-1], False, "step size underflow", step, arc,
                    np.array(path),
                )

        if w[-1] >= 1.0:
            # Crossed the target: interpolate to lam = 1, then polish.
            s = (1.0 - y[-1]) / (w[-1] - y[-1])
            end = _newton_at_end(H, J, y + s * (w - y), end_tol, 50)
            arc += np.linalg.norm(end - y)
            path.append(end)
            return TrackResult(
                end[:-1], 1.0, True, "reached lam = 1", step, arc, np.array(path)
            )

        arc += np.linalg.norm(w - y)
        y = w
        path.append(y.copy())
        if arc > max_arc:
            return TrackResult(
                y[:-1], y[-1], False, "arc length limit exceeded", step, arc,
                np.array(path),
            )
        # Grow the step when the corrector converged easily.
        if iters <= 2:
            h = min(h_max, 2.0 * h)
        elif iters >= max_corrector - 1:
            h *= 0.5

    return TrackResult(
        y[:-1], y[-1], False, "step limit exceeded", max_steps, arc, np.array(path)
    )
