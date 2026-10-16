"""Homotopy and Newton-type solvers for nonlinear complementarity problems."""

from . import equations, functions, ncp, problems
from .ncp import NCPResult, homotopy, semismooth_newton, smoothing_newton, solve
from .tracking import TrackResult, track

__all__ = [
    "NCPResult",
    "TrackResult",
    "equations",
    "functions",
    "homotopy",
    "ncp",
    "problems",
    "semismooth_newton",
    "smoothing_newton",
    "solve",
    "track",
]

__version__ = "0.1.0"
