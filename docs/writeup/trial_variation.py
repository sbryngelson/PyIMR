"""Is the residual structure trial-to-trial parameter variation, or model error?

The two have been conflated. A hierarchical model says each bubble has its own parameters
theta_j ~ N(theta_pop, Sigma), so the mean trace has covariance

    C = (1/J) * G Sigma G^T + (sigma_meas^2 / J) I,     G = d(R/R_max)/d theta,

low rank plus diagonal rather than diagonal. If the lag-one correlation of 0.918 in the fit
residual comes from that structure, whitening by C removes it and the model is adequate. If
it survives whitening, it is structure the model genuinely lacks. That is the test.
"""
import os, json
from pathlib import Path
for _n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_n,"1")
import numpy as np

R_MAX, STRETCH = 277e-6, 7.09
FIT = (204.3, 0.04651, 1.964e-7, 5.301)
PATHS = ("R0", "Req", "material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
VALUES = np.array([R_MAX, R_MAX/STRETCH, FIT[0], FIT[1], FIT[2], FIT[3]])

def lag_one(x):
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))

def main():
    import pyimr
    record = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))["gelatin_15C"]
    times = np.array(record["times_s"]); mean = np.array(record["mean"]); spread = np.array(record["spread"])
    keep = spread > 0
    times, mean, spread = times[keep], mean[keep], spread[keep]
    trials = record["trials"]

    config = pyimr.SimulationConfig(R_MAX, R_MAX/STRETCH,
        pyimr.QuadraticZener(FIT[0], FIT[1], FIT[2], 0.0, FIT[3]),
        dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=400_000)
    problem = pyimr.prepare(config)
    model = np.asarray(problem.solve(times).radius_ratio, dtype=float)
    # d(R/R_max)/d log theta: scaling by the value makes Sigma dimensionless
    G = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio, dtype=float) * VALUES
    print(f"{len(times)} samples, {trials} trials, sensitivity span {G.shape[1]}-dimensional\n")

    # --- how much of each trial's deviation lies in span(G)? ---
    DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data/Ga_t15_exp_data.csv"
    table = np.loadtxt(DATA, delimiter=",", ndmin=2)
    raw = table[:, 1:].T
    good = ~(raw > 1.05).any(axis=0) & (raw.std(axis=0, ddof=1) > 0.0)
    deviations = (raw[:, good] - raw[:, good].mean(axis=0))[:, keep] if good.sum() != keep.size else None
    if deviations is None or deviations.shape[1] != len(times):
        sel = raw[:, good]; sel = sel - sel.mean(axis=0)
        deviations = sel[:, :len(times)] if sel.shape[1] >= len(times) else None
    basis, _ = np.linalg.qr(G)
    explained = deviations @ basis @ basis.T
    leftover = deviations - explained
    share = 1.0 - (leftover**2).sum() / (deviations**2).sum()
    print(f"PARAMETER VARIATION explains {share:.1%} of the total trial-to-trial variance")
    print(f"  chance level for a {G.shape[1]}-dimensional span: {G.shape[1]/len(times):.1%}")
    print(f"  lag-one of a typical deviation  : {np.median([lag_one(d) for d in deviations]):.3f}")
    print(f"  lag-one after removing the span : {np.median([lag_one(d) for d in leftover]):.3f}\n")

    # Sigma itself is not estimable: the g-alpha degeneracy makes the regression onto G
    # ill-conditioned, and solving for coefficients gives sd(log g) = 159. But only the
    # PREDICTION covariance G Sigma G^T enters the likelihood, and that is the sample
    # covariance of the projected deviations -- no inversion, no degeneracy.
    low_rank = explained.T @ explained / (deviations.shape[0] - 1)
    noise = float((leftover**2).sum() / leftover.size)
    values = np.linalg.eigvalsh(low_rank)[::-1]
    print(f"prediction covariance of the parameter part: rank {int((values > 1e-12*values[0]).sum())},")
    print(f"  leading eigenvalues {np.array2string(np.sqrt(np.maximum(values[:4],0)), precision=5)}")
    print(f"  leftover per-sample sd: {np.sqrt(noise):.5f} of R_max\n")

    # --- the test: does the fit residual whiten under C? ---
    residual = model - mean
    C = (low_rank + noise*np.eye(len(times))) / trials
    C += 1e-14*np.eye(len(times))
    factor = np.linalg.cholesky(C)
    whitened = np.linalg.solve(factor, residual)
    diagonal = residual / (spread/np.sqrt(trials))
    print("THE TEST -- fit residual of qSLS against the mean trace")
    print(f"  {'':22} {'chi2/N':>9} {'lag-one':>9}")
    print(f"  {'diagonal (i.i.d.)':22} {float(diagonal@diagonal)/len(times):9.3f} {lag_one(residual):9.3f}")
    print(f"  {'hierarchical C':22} {float(whitened@whitened)/len(times):9.3f} {lag_one(whitened):9.3f}")
    print("\n  if the hierarchical lag-one is near zero, the correlation WAS parameter")
    print("  variation; if it survives, it is structure the model does not contain.")
    print(f"\n  note the two chi2/N differ by ~{trials}x from the SAME residual: the document")
    print(f"  uses the trial-to-trial spread as the error on the MEAN of {trials} trials,")
    print(f"  where the standard error of that mean is smaller by sqrt({trials}) = {np.sqrt(trials):.2f}.")
    plain = float((residual/spread) @ (residual/spread))/len(times)
    print(f"    chi2/N with spread as the error : {plain:.3f}   (what the tables report)")
    print(f"    chi2/N with the standard error  : {plain*trials:.3f}")

if __name__ == "__main__":
    main()
