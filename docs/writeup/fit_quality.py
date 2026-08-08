"""How good are the fits, measured against what the residuals actually are?

chi^2/N near one is the headline number, but it presumes the residuals are independent
draws of known scale. Neither holds here: they are strongly autocorrelated, and part of
the spread used as the denominator is variation in the parameters across trials rather
than measurement error. Both are quantified alongside the fit so the headline can be read
with the right amount of confidence.
"""
import os, json
for _n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_n,"1")
import numpy as np


R_MAX, DATASET = 277e-6, "gelatin_15C"
BOX = {"g": (1e2, 1e5), "mu": (1e-4, 1.0), "lam": (1e-9, 1e-3), "alpha": (1e-3, 1e2)}
MODELS = {
  "qSLS": (("material.shear_modulus_pa","material.viscosity_pa_s","material.relaxation_time_s","material.stiffening"),
           ("g","mu","lam","alpha"), (204.3, 0.04651, 1.964e-7, 5.301)),
  "SLS":  (("material.shear_modulus_pa","material.viscosity_pa_s","material.relaxation_time_s"),
           ("g","mu","lam"), (2154.0, 0.0464, 2.783e-7)),
  "qKV":  (("material.shear_modulus_pa","material.viscosity_pa_s","material.stiffening"),
           ("g","mu","alpha"), (2154.0, 0.0464, 0.4642)),
  "NHKV": (("material.shear_modulus_pa","material.viscosity_pa_s"),
           ("g","mu"), (2500.0, 0.1)),
}
STARTS = 8

def _material(name, v):
    import pyimr
    if name == "qSLS": return pyimr.QuadraticZener(v[0], v[1], v[2], 0.0, v[3])
    if name == "SLS":  return pyimr.Zener(v[0], v[1], v[2], 0.0)
    if name == "qKV":  return pyimr.QuadraticKelvinVoigt(v[0], v[1], v[2])
    return pyimr.NeoHookeanKelvinVoigt(v[0], v[1])

def _job(name):
    import pyimr
    from scipy.optimize import least_squares
    from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
    record = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))[DATASET]
    times = np.array(record["times_s"]); mean = np.array(record["mean"]); spread = np.array(record["spread"])
    keep = spread > 0
    times, mean, spread = times[keep], mean[keep], spread[keep]
    paths, keys, start_values = MODELS[name]
    lo = np.array([BOX[k][0] for k in keys]); hi = np.array([BOX[k][1] for k in keys])
    config = pyimr.SimulationConfig(R_MAX, R_MAX/record["stretch"], _material(name, start_values),
                                    dynamics="keller-miksis", rtol=1e-8, atol=1e-10, max_steps=300_000)
    inference = prepare_inference(config, RadiusObservation(times, mean*R_MAX, spread*R_MAX),
        tuple(InferenceParameter(p, a, b, "log") for p, a, b in zip(paths, lo, hi)))
    default = np.clip((np.log(start_values)-np.log(lo))/(np.log(hi)-np.log(lo)), 1e-4, 1-1e-4)
    rng = np.random.default_rng(0)
    starts = np.vstack([default, rng.random((STARTS-1, len(keys)))])
    best, residual = np.inf, None
    for s in starts:
        try:
            got = least_squares(inference.residual, np.clip(s,1e-6,1-1e-6), jac=inference.jacobian,
                                bounds=(0.0,1.0), x_scale="jac", max_nfev=300)
        except Exception: continue                                       # noqa: BLE001
        value = float(got.fun @ got.fun)
        if value < best: best, residual = value, np.asarray(got.fun, float)
    return name, best, residual, len(times)

def main():
    from pyimr.parallel import worker_pool
    from pyimr.noise import check_residuals
    with worker_pool(4) as pool:
        results = list(pool.map(_job, list(MODELS)))
    print(f"{DATASET}: fits by multistart Gauss-Newton with analytic Jacobians, "
          f"{STARTS} starts, wide physical prior box\n")
    print(f"{'model':>6} {'params':>7} {'chi2/N':>9} {'lag-1 rho':>10} {'N_eff':>7} "
          f"{'chi2/N_eff':>11} {'inflation':>10}")
    store = {}
    for name, chi2, residual, n in results:
        if residual is None: print(f"{name:>6}   no usable fit"); continue
        check = check_residuals(residual)
        store[name] = dict(chi2_per_n=chi2/n, lag1=check.lag_one, n_eff=check.effective_samples,
                           inflation=check.inflation, params=len(MODELS[name][1]), n=n)
        print(f"{name:>6} {len(MODELS[name][1]):7d} {chi2/n:9.3f} {check.lag_one:10.3f} "
              f"{check.effective_samples:7.1f} {chi2/max(check.effective_samples,1e-9):11.2f} "
              f"{check.inflation:10.2f}")
    print(f"\n  chi2/N is measured against {results[0][3]} samples, but the residuals are")
    print("  autocorrelated: the effective count is what a chi-squared test actually has.")
    json.dump(store, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitquality.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
