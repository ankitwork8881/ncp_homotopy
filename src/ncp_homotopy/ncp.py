"""Solvers for the nonlinear complementarity problem.

NCP(f): find x in R^n with

    x >= 0,   f(x) >= 0,   x_i f_i(x) = 0  for all i.

Three methods are implemented:

``semismooth_newton``
    Generalized Newton method on the Fischer-Burmeister reformulation,
    globalized with an Armijo line search on the merit function
    0.5 ||Phi(x)||^2 and a steepest-descent fallback.

``smoothing_newton``
    Newton's method on the smoothed Fischer-Burmeister system for a
    decreasing sequence of smoothing parameters mu.

``homotopy``
    Fixed-point homotopy on the smoothed system in z = (x, y), where y is
    a slack for f(x), with the smoothing parameter tied to the homotopy
    parameter, mu = beta (1 - lam). At lam = 1 the system is the exact
    min-function reformulation; the end point is polished with the
    semismooth Newton method.
"""

from dataclasses import dataclass

import numpy as np

from . import functions as fn
from .tracking import track


@dataclass
class NCPResult:
    """Approximate NCP solution and its quality measures."""

    x: np.ndarray
    fx: np.ndarray
    success: bool
    method: str
    iterations: int
    residual: float
    gap: float


def _result(f, x, method, iterations, tol):
    fx = np.asarray(f(x), dtype=float)
    residual = fn.natural_residual(x, fx)
    return NCPResult(
        x, fx, residual <= tol, method, iterations, residual,
        fn.complementarity_gap(x, fx),
    )


def semismooth_newton(
    f, jf, x0, *, tol=1e-10, max_iter=200, sigma=1e-4, beta=0.5, rho=1e-10, p=2.1
):
    """Semismooth Newton method for NCP(f) on the Fischer-Burmeister system."""
    x = np.asarray(x0, dtype=float).copy()

    def merit(z):
        r = fn.phi_fb(z, f(z))
        return 0.5 * float(r @ r)

    for k in range(max_iter):
        fx = f(x)
        if fn.natural_residual(x, fx) <= tol:
            return _result(f, x, "semismooth_newton", k, tol)

        phi = fn.phi_fb(x, fx)
        da, db = fn.phi_fb_grad(x, fx)
        V = np.diag(da) + db[:, None] * jf(x)
        grad = V.T @ phi

        try:
            d = np.linalg.solve(V, -phi)
            if grad @ d > -rho * np.linalg.norm(d) ** p:
                d = -grad
        except np.linalg.LinAlgError:
            d = -grad

        psi = 0.5 * float(phi @ phi)
        slope = float(grad @ d)
        t = 1.0
        while merit(x + t * d) > psi + sigma * t * slope and t > 1e-14:
            t *= beta
        x = x + t * d

    return _result(f, x, "semismooth_newton", max_iter, tol)


def smoothing_newton(
    f, jf, x0, *, mu0=1.0, shrink=0.1, mu_min=1e-14, tol=1e-10, inner_iter=50
):
    """Continuation in mu on the smoothed Fischer-Burmeister system."""
    x = np.asarray(x0, dtype=float).copy()
    mu = mu0
    total = 0

    while True:
        for _ in range(inner_iter):
            fx = f(x)
            phi = fn.phi_fb_smooth(x, fx, mu)
            if np.linalg.norm(phi, np.inf) <= max(mu, tol):
                break
            da, db = fn.phi_fb_smooth_grad(x, fx, mu)
            V = np.diag(da) + db[:, None] * jf(x)
            step = np.linalg.lstsq(V, -phi, rcond=None)[0]
            # Damp the step until the smoothed residual decreases.
            t, base = 1.0, np.linalg.norm(phi)
            while t > 1e-10 and np.linalg.norm(
                fn.phi_fb_smooth(x + t * step, f(x + t * step), mu)
            ) >= base:
                t *= 0.5
            x = x + t * step
            total += 1
        if mu <= mu_min or fn.natural_residual(x, f(x)) <= tol:
            break
        mu *= shrink

    return _result(f, x, "smoothing_newton", total, tol)


def smoothed_homotopy(f, jf, a, beta=1.0):
    """Return (H, J) for the smoothed fixed-point homotopy in z = (x, y).

    G_mu(z) = [f(x) - y,  phi_chks(x, y, mu)],   mu = beta (1 - lam),
    H(z, lam) = lam G_mu(z) + (1 - lam)(z - a).
    """
    a = np.asarray(a, dtype=float)
    n = a.size // 2
    eye_n = np.eye(n)
    eye_2n = np.eye(2 * n)

    def G(z, mu):
        x, y = z[:n], z[n:]
        return np.concatenate([f(x) - y, fn.phi_chks(x, y, mu)])

    def H(w):
        z, lam = w[:-1], w[-1]
        return lam * G(z, beta * (1.0 - lam)) + (1.0 - lam) * (z - a)

    def J(w):
        z, lam = w[:-1], w[-1]
        mu = beta * (1.0 - lam)
        x, y = z[:n], z[n:]
        dx, dy, dmu = fn.phi_chks_grad(x, y, mu)
        dG = np.block([[jf(x), -eye_n], [np.diag(dx), np.diag(dy)]])
        dz = lam * dG + (1.0 - lam) * eye_2n
        dG_dmu = np.concatenate([np.zeros(n), dmu])
        dlam = G(z, mu) - lam * beta * dG_dmu - (z - a)
        return np.column_stack([dz, dlam])

    return H, J


def homotopy(f, jf, x0, *, beta=1.0, tol=1e-10, polish=True, **track_options):
    """Solve NCP(f) by tracking the smoothed fixed-point homotopy.

    The start is z0 = (x0, f(x0)). Extra keyword arguments go to
    :func:`tracking.track`.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    a = np.concatenate([x0, f(x0)])
    H, J = smoothed_homotopy(f, jf, a, beta)
    path = track(H, J, a, **track_options)
    x = path.u[:n]
    iterations = path.steps
    if polish:
        polished = semismooth_newton(f, jf, x, tol=tol)
        x, iterations = polished.x, iterations + polished.iterations
    result = _result(f, x, "homotopy", iterations, tol)
    result.success = result.success and path.success
    return result


METHODS = {
    "semismooth_newton": semismooth_newton,
    "smoothing_newton": smoothing_newton,
    "homotopy": homotopy,
}


def solve(f, jf, x0, method="homotopy", **options):
    """Solve NCP(f) with the named method; see :data:`METHODS`."""
    try:
        solver = METHODS[method]
    except KeyError:
        raise ValueError(f"unknown NCP method {method!r}") from None
    return solver(f, jf, x0, **options)
