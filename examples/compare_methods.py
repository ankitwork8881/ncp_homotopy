"""Compare the NCP solvers and the two equation homotopies on the bundled
test problems.

    python examples/compare_methods.py
"""

import time

import numpy as np

from ncp_homotopy import equations, ncp, problems


def equations_table():
    print("Nonlinear equations F(x) = 0")
    print(f"{'problem':<24}{'homotopy':<13}{'ok':<6}{'steps':>7}{'arc':>10}{'|F(x)|':>12}")
    for p in [
        problems.narrow_valley(),
        problems.freudenstein_roth(),
        problems.cubic_with_coupling(5),
    ]:
        for method in ["fixed_point", "newton"]:
            r = equations.solve(p.f, p.jac, p.start, method=method, max_arc=200)
            print(
                f"{p.name:<24}{method:<13}{str(r.success):<6}{r.steps:>7}"
                f"{r.arc_length:>10.2f}{np.linalg.norm(p.f(r.u)):>12.2e}"
            )


def ncp_table():
    print("\nComplementarity problems NCP(f)")
    print(f"{'problem':<20}{'method':<20}{'ok':<6}{'iter':>6}{'residual':>12}{'time (ms)':>11}")
    for p in [
        problems.random_pd_lcp(10),
        problems.kojima_shindo(),
        problems.monotone_cubic_ncp(),
    ]:
        for method in ncp.METHODS:
            start = time.perf_counter()
            r = ncp.solve(p.f, p.jac, p.start, method=method)
            elapsed = 1000 * (time.perf_counter() - start)
            print(
                f"{p.name:<20}{method:<20}{str(r.success):<6}{r.iterations:>6}"
                f"{r.residual:>12.2e}{elapsed:>11.1f}"
            )


if __name__ == "__main__":
    equations_table()
    ncp_table()
