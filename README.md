# ncp_homotopy

Homotopy continuation and Newton-type solvers for the **nonlinear complementarity problem** (NCP), written in Python 3 with NumPy only.

Given `f: Rⁿ → Rⁿ`, the NCP asks for `x` with

```
x ≥ 0,    f(x) ≥ 0,    xᵢ fᵢ(x) = 0   for every i.
```

NCPs come up in equilibrium models, contact mechanics, and as the KKT conditions of bound-constrained optimization. The usual approach rewrites the problem as a system of equations using an NCP function `φ(a, b)`, which is zero exactly when `a ≥ 0, b ≥ 0, ab = 0`. That system is nonsmooth, so you can either use a generalized Newton method on it directly, smooth it, or follow a homotopy path to its solution.

## What's inside

| Module | Contents |
|---|---|
| `functions` | NCP functions (min, Fischer–Burmeister), their smoothings (CHKS, smoothed FB), generalized gradients at the kinks, residual measures |
| `tracking` | Predictor–corrector path tracker: null-space tangent, Euler predictor, minimum-norm (normal flow) corrector, adaptive step, end game at λ = 1 |
| `equations` | Fixed-point and Newton homotopies for `F(x) = 0` |
| `ncp` | Three NCP solvers: `semismooth_newton`, `smoothing_newton`, `homotopy` |
| `problems` | Test problems with known solutions or properties |

### NCP solvers

- **Semismooth Newton.** Generalized Newton on `Φ(x) = φ_FB(x, f(x)) = 0`, globalized with an Armijo search on `½‖Φ‖²` and a steepest-descent fallback when the Newton direction isn't a good descent direction.
- **Smoothing Newton.** Newton on the smoothed FB system for `μ = 1, 0.1, 0.01, …`, warm-starting each level from the previous one.
- **Homotopy.** Introduces a slack `y = f(x)` and tracks
  `H(z, λ) = λ G_μ(z) + (1 − λ)(z − a)`, with `z = (x, y)`, `G_μ(z) = [f(x) − y, φ_CHKS,μ(x, y)]` and `μ = β(1 − λ)`.
  Smoothing and continuation are driven by a single parameter. At λ = 0 the zero is the start point. At λ = 1, μ = 0, so the end point solves the exact min-function reformulation. That end point is then polished with semismooth Newton.

## Install and run

```bash
python -m pip install -e ".[test]"
python -m pytest
python examples/compare_methods.py
```

```python
import numpy as np
from ncp_homotopy import ncp, problems

p = problems.kojima_shindo()
result = ncp.solve(p.f, p.jac, p.start, method="homotopy")
print(result.x, result.residual)   # [1.2247 0. 0. 0.5]  ~1e-16
```

## Results

Output of `examples/compare_methods.py`:

```
Nonlinear equations F(x) = 0
problem                 homotopy     ok      steps       arc      |F(x)|
narrow_valley           fixed_point  True       53      2.52    0.00e+00
narrow_valley           newton       True      236     14.03    1.12e-27
freudenstein_roth       fixed_point  True      109     10.84    0.00e+00
freudenstein_roth       newton       True      112     70.80    8.88e-16
cubic_with_coupling_5   fixed_point  True       31      2.24    4.05e-13
cubic_with_coupling_5   newton       True       24      2.29    6.66e-15

Complementarity problems NCP(f)
problem             method              ok      iter    residual
lcp                 semismooth_newton   True      10    6.66e-16
lcp                 smoothing_newton    True      10    2.86e-12
lcp                 homotopy            True     151    8.88e-16
kojima_shindo       semismooth_newton   True       8    2.14e-16
kojima_shindo       smoothing_newton    True       8    2.89e-16
kojima_shindo       homotopy            True     134    1.52e-16
monotone_cubic_6    semismooth_newton   True       7    8.88e-16
monotone_cubic_6    smoothing_newton    True       7    1.71e-12
monotone_cubic_6    homotopy            True     127    6.66e-16
```

Observations:

- **Narrow valley.** `f(x) = arctan(100x)/π + sin(5x/(x²+0.2))/2 + 0.1x` has a single root at 0, but `½f²` has spurious local minima. Starting at `x = 1`, damped Newton stalls at `x ≈ 0.2423`, which a test checks. Both homotopies reach the root. The fixed-point homotopy gets there along a much shorter path (arc length 2.5 versus 14).
- **Freudenstein–Roth.** Newton's method stalls near a local minimiser of the residual. The fixed-point path to the root `(5, 4)` is about 7× shorter than the Newton-homotopy path.
- **NCPs.** Newton-type methods are much cheaper when they converge. The homotopy costs more per solve but doesn't rely on a merit function to make progress. On the Kojima–Shindo problem all three methods reach the degenerate solution `(√6/2, 0, 0, ½)`.

## Tests

The suite in `tests/` checks:

- the defining property of each NCP function, plus the smoothing error bounds `|φ_CHKS,μ − φ_min| ≤ 2μ` and `|φ_FB,μ − φ_FB| ≤ √2 μ`;
- every analytic gradient and the full homotopy Jacobian against central differences;
- that the tracker's tangent spans the Jacobian null space and keeps its orientation;
- each solver on random positive-definite LCPs against an exact solution found by enumerating supports;
- a monotone nonlinear NCP, where all methods must agree on the unique solution;
- the Kojima–Shindo problem, where the result must be one of the two known solutions.

## License

MIT. See `LICENSE`.
