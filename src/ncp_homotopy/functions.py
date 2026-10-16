"""NCP functions, their smoothings, and residual measures.

A function phi: R^2 -> R is an *NCP function* if

    phi(a, b) = 0  <=>  a >= 0, b >= 0, a * b = 0.

Applying phi componentwise to (x_i, f_i(x)) turns the complementarity
problem into a square system of (generally nonsmooth) equations. All
functions here are vectorised over numpy arrays.
"""

import numpy as np

# Element of the generalized Jacobian used for the Fischer-Burmeister
# function at its single kink (a, b) = (0, 0).
_FB_KINK = 1.0 - 1.0 / np.sqrt(2.0)


def phi_min(a, b):
    """Min function written as a + b - |a - b| (= 2 min(a, b))."""
    return a + b - np.abs(a - b)


def phi_fb(a, b):
    """Fischer-Burmeister function a + b - sqrt(a^2 + b^2)."""
    return a + b - np.hypot(a, b)


def phi_fb_grad(a, b):
    """Partial derivatives (d/da, d/db) of phi_fb.

    At the origin, where phi_fb is not differentiable, an element of the
    Clarke generalized gradient is returned instead.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.hypot(a, b)
    kink = r == 0.0
    safe_r = np.where(kink, 1.0, r)
    da = np.where(kink, _FB_KINK, 1.0 - a / safe_r)
    db = np.where(kink, _FB_KINK, 1.0 - b / safe_r)
    return da, db


def phi_chks(a, b, mu):
    """Chen-Harker-Kanzow-Smale smoothing of phi_min.

    a + b - sqrt((a - b)^2 + 4 mu^2). Smooth for mu != 0 and equal to
    phi_min at mu = 0. The approximation error is at most 2|mu|.
    """
    return a + b - np.sqrt((a - b) ** 2 + 4.0 * mu ** 2)


def phi_chks_grad(a, b, mu):
    """Partial derivatives (d/da, d/db, d/dmu) of phi_chks.

    At mu = 0 and a = b the function has a kink; the returned values are
    then the generalized-gradient element (1, 1, 0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.sqrt((a - b) ** 2 + 4.0 * mu ** 2)
    kink = r == 0.0
    safe_r = np.where(kink, 1.0, r)
    s = np.where(kink, 0.0, (a - b) / safe_r)
    dmu = np.where(kink, 0.0, -4.0 * mu / safe_r)
    return 1.0 - s, 1.0 + s, dmu


def phi_fb_smooth(a, b, mu):
    """Smoothed Fischer-Burmeister function a + b - sqrt(a^2 + b^2 + 2 mu^2)."""
    return a + b - np.sqrt(a ** 2 + b ** 2 + 2.0 * mu ** 2)


def phi_fb_smooth_grad(a, b, mu):
    """Partial derivatives (d/da, d/db) of phi_fb_smooth for mu != 0."""
    r = np.sqrt(a ** 2 + b ** 2 + 2.0 * mu ** 2)
    return 1.0 - a / r, 1.0 - b / r


def natural_residual(x, fx):
    """Infinity norm of min(x, f(x)); zero exactly at an NCP solution."""
    return float(np.max(np.abs(np.minimum(x, fx)), initial=0.0))


def complementarity_gap(x, fx):
    """Absolute value of x^T f(x)."""
    return float(abs(np.dot(x, fx)))


def is_ncp_solution(x, fx, tol=1e-8):
    """Check feasibility (x >= 0, f(x) >= 0) and complementarity to ``tol``."""
    x = np.asarray(x, dtype=float)
    fx = np.asarray(fx, dtype=float)
    return bool(
        np.all(x >= -tol)
        and np.all(fx >= -tol)
        and np.all(np.abs(x * fx) <= tol * (1.0 + np.abs(x) + np.abs(fx)))
    )
