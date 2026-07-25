"""
Standalone validation of thermal_fd.finite_diff_mat against closed-form test
functions, BEFORE wiring anything into the ODE state. No IMRv2 round-trip
needed for this stage -- these are exact analytic checks.

Exactness class of the interior spherical-Laplacian stencil: by Taylor
expansion, its leading truncation error is
    h^2 * [ f''''(y)/12 + f'''(y)/(3y) ]
which vanishes identically for quadratics-and-below (f'''=f''''=0) but NOT
for quartics -- so quadratic test functions get machine-precision checks,
and a quartic is used only as a DOCUMENTED, hand-predicted O(h^2) check
(not a pass/fail against a tight tolerance), to make the truncation-error
behaviour explicit rather than hiding it behind a loose tolerance.
"""
import numpy as np
from thermal_fd import finite_diff_mat

PASS = "PASS"
FAIL = "FAIL"


def check(label, err, tol):
    tag = PASS if err < tol else FAIL
    print(f"    {label:42s} err={err:.3e}  tol={tol:.1e}  {tag}")
    return err < tol


ok = True

print("=" * 72)
print("1. INTERIOR grid (tm_check=0): spherical Laplacian, order=2")
print("   EXACTNESS check, f(y) = 1 + 2y^2  ->  Lap f = 12  (f'''=f''''=0,")
print("   so the stencil is exact to machine precision for this class)")
print("=" * 72)
for nodes in [11, 25, 51, 101]:
    N = nodes - 1
    y = np.linspace(0, 1, nodes)
    f = 1 + 2 * y**2
    lap_exact = 12.0
    D2 = finite_diff_mat(nodes, 2, tm_check=0)
    lap_num = D2 @ f
    err_interior = np.max(np.abs(lap_num[1:N] - lap_exact))
    err_center = abs(lap_num[0] - lap_exact)
    ok &= check(f"nodes={nodes:4d} interior (machine precision)", err_interior, 1e-9)
    ok &= check(f"nodes={nodes:4d} row-0 center (machine precision)", err_center, 1e-9)

print()
print("=" * 72)
print("1b. Same stencil, quartic f(y) = 1+2y^2+3y^4  ->  Lap f = 12+60y^2")
print("    NOT exact for quartics (f''''!=0) -- error should match the")
print("    HAND-DERIVED leading truncation term h^2*30 (interior) and")
print("    (h^2/4)*72 = 18h^2 (row 0), not machine precision.")
print("=" * 72)
for nodes in [25, 51, 101]:
    N = nodes - 1
    h = 1.0 / N
    y = np.linspace(0, 1, nodes)
    f = 1 + 2 * y**2 + 3 * y**4
    lap_exact = 12 + 60 * y**2
    D2 = finite_diff_mat(nodes, 2, tm_check=0)
    lap_num = D2 @ f
    err_interior = np.max(np.abs(lap_num[1:N] - lap_exact[1:N]))
    err_center = abs(lap_num[0] - lap_exact[0])
    pred_interior = 30 * h**2
    pred_center = 18 * h**2
    ok &= check(f"nodes={nodes:4d} interior vs HAND-PREDICTED {pred_interior:.4e}",
                abs(err_interior - pred_interior), 1e-4 * pred_interior + 1e-8)
    ok &= check(f"nodes={nodes:4d} row-0 vs HAND-PREDICTED {pred_center:.4e}",
                abs(err_center - pred_center), 1e-4 * pred_center + 1e-8)

print()
print("=" * 72)
print("2. CONVERGENCE ORDER, non-polynomial regular function")
print("   f(y) = exp(-y^2)  ->  Lap f = exp(-y^2)*(4y^2 - 6)  (closed form)")
print("=" * 72)
errs = []
Ns = [21, 41, 81, 161]
for nodes in Ns:
    N = nodes - 1
    y = np.linspace(0, 1, nodes)
    f = np.exp(-y**2)
    lap_exact = np.exp(-y**2) * (4 * y**2 - 6)
    D2 = finite_diff_mat(nodes, 2, tm_check=0)
    lap_num = D2 @ f
    err = np.max(np.abs(lap_num[0:N] - lap_exact[0:N]))
    errs.append(err)
    print(f"    nodes={nodes:4d}  max err (incl. center) = {err:.4e}")
rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
print(f"    observed convergence rates: {[f'{r:.2f}' for r in rates]}  (expect ~2.0)")
ok &= all(r > 1.9 for r in rates)
print(f"    2nd-order convergence: {PASS if all(r > 1.9 for r in rates) else FAIL}")

print()
print("=" * 72)
print("3. INTERIOR grid: first derivative, order=1")
print("   EXACTNESS check, f(y) = 1 + 2y^2  ->  f' = 4y")
print("   (central diff is exact for quadratics: error ~ f'''=0)")
print("=" * 72)
for nodes in [25, 51]:
    N = nodes - 1
    y = np.linspace(0, 1, nodes)
    f = 1 + 2 * y**2
    fp_exact = 4 * y
    D1 = finite_diff_mat(nodes, 1, tm_check=0)
    fp_num = D1 @ f
    err_interior = np.max(np.abs(fp_num[1:N] - fp_exact[1:N]))
    ok &= check(f"nodes={nodes:4d} interior (central diff, machine prec.)", err_interior, 1e-9)
    err_wall = abs(fp_num[N] - fp_exact[N])
    ok &= check(f"nodes={nodes:4d} wall (one-sided, machine prec.)", err_wall, 1e-9)
    row0_is_zero = np.allclose(D1[0, :], 0.0)
    print(f"    nodes={nodes:4d} row 0 (center) identically zero: "
          f"{PASS if row0_is_zero else FAIL}")
    ok &= row0_is_zero

print()
print("=" * 72)
print("4. EXTERIOR grid (tm_check=1): plain (non-spherical) stencils")
print("   g(xi) = xi^2  ->  d/dxi = 2xi,  d2/dxi2 = 2  (domain descends 1 -> -1)")
print("=" * 72)
for nodes in [25, 51]:
    N = nodes - 1
    deltaY = -2.0 / N
    xi = 1.0 + np.arange(nodes) * deltaY
    g = xi**2
    D1 = finite_diff_mat(nodes, 1, tm_check=1)
    D2 = finite_diff_mat(nodes, 2, tm_check=1)
    gp_num = D1 @ g
    gpp_num = D2 @ g
    err1 = np.max(np.abs(gp_num[1:N] - 2 * xi[1:N]))
    err2 = np.max(np.abs(gpp_num[1:N] - 2.0))
    ok &= check(f"nodes={nodes:4d} d/dxi interior", err1, 1e-8)
    ok &= check(f"nodes={nodes:4d} d2/dxi2 interior", err2, 1e-8)

print()
print("=" * 72)
print("ALL CHECKS:", PASS if ok else FAIL)
print("=" * 72)
