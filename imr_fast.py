"""Fast Python IMR solver -- a validated slice of IMRv2.

Covers:
  radial   1 (Rayleigh-Plesset), 2 (Keller-Miksis, pressure form),
           3 (Keller-Miksis, enthalpy form, Tait EoS liquid),
           4 (Gilmore, Tait EoS liquid),
           5 (Keller-Miksis, enthalpy form, Mie-Gruneisen EoS liquid)
           radial=6 (Gilmore, Mie-Gruneisen EoS) NOT supported -- confirmed
           dead/broken upstream (goes complex in real IMRv2 for essentially
           every polytropic-gas P, mild or extreme; f_radial_eq.m evaluates
           the EoS at raw P with no pressure-reference correction, unlike
           radial=5's Pb=P-iWe/R)
  bubtherm 0 (polytropic, kappa) or 1 (gas thermal PDE)
           medtherm 0 or 1 (liquid boundary layer, requires bubtherm=1)
           masstrans 0 or 1 (vapor mass transfer, requires bubtherm=1 and
                             vapor=1; may be combined with medtherm=1)
  stress   0 (no stress)
           1 (neo-Hookean Kelvin-Voigt)
           2 (quadratic Kelvin-Voigt, strain-stiffening, alphax)
           3 (linear Maxwell / Jeffreys / Zener, 1 internal variable)
           4 (linear Maxwell / Jeffreys / Zener, quadratic KV elastic term,
              1 internal variable -- stress=3's counterpart to stress=2)
           5 (UCM / Oldroyd-B, 2 internal variables)
  forcing  wave_type 0 (constant offset), 1 (Gaussian), 2 (histotripsy),
           3 (Heaviside step), or a dimensional sampled pressure history

Equations transcribed from IMRv2/src/{f_radial_eq,f_stress,f_call_params,
f_imr_fd}.m. Validated against IMRv2 reference trajectories -- see
tests/run_validation.py.

The default physical constants match the pinned reference trajectories.
They are configurable through PhysicalParameters; IMRv2's own shipped
polytropic exponent is 1.47, whereas this solver defaults to 1.4 because that
is the value used to generate its reference data.

bubtherm=1 implements IMRv2's "elseif bubtherm" branch (f_imr_fd.m): gas-phase
thermal PDE, dry gas (kv0=0, vapor=0). With medtherm=0, the wall is an
isothermal-equivalent clamp (thetadot[-1]=0). Its Pdot uses bare P (kappa*P),
NOT (P-Pv) -- this is IMRv2's actual equation for this branch, not a
simplification; the bubtherm=0 polytropic branch's Pdot uses (P-Pv) and the
two are NOT reconciled/harmonized, since they are genuinely different
equations in the source.

medtherm=1 (requires bubtherm=1) adds the liquid boundary layer: a stretched
exterior grid (Mt points, Lt controls the stretching), advection+diffusion+
viscous-dissipation RHS for Tm, and the wall temperature theta[-1] is NOT a
free state -- it is solved every RHS call via a 1-D root-find (scipy.optimize
style secant iteration; f_bubble_wall_thermal_bc) enforcing heat-flux
continuity across the interface. A warm-start value is scoped to one
integration, matching IMRv2's continuation behavior without leaking state
between repeated prepared solves.
thetadot[-1]=0 and Tmdot[0]=0 always because both slots are algebraic boundary
values rather than evolved states. GRADIENTS ARE NOT VALIDATED for medtherm=1:
naive autodiff through the root-find would need the implicit function theorem,
not attempted here.

masstrans=1 (requires bubtherm=1, vapor=1) implements IMRv2's "if bubtherm &&
masstrans" branch: a wall vapor mass fraction field kv(y,t), a mixture
thermal conductivity/diffusivity (kv-weighted gas/vapor), extra mass-transfer
terms in Pdot/Uvel/thetadot, and a new kvdot equation. kv[-1] (the wall value)
is set algebraically every RHS call via vapor-liquid equilibrium (_kv_of_T,
f_kv_of_T.m) using T[-1] computed with the STALE (pre-update) kv[-1] --
IMRv2's own one-step lag, replicated exactly, not reconciled. With medtherm=0,
theta[-1] never evolves (frozen at its initial value), so T[-1]===1
identically and no implicit solve is needed for the wall BC itself. With
medtherm=1 also set, theta[-1] is instead solved via a 3-term coupled
root-find (_wall_theta_bw_full, f_bubble_wall_full_bc) that additionally
enforces vapor-mass-flux continuity (via a preliminary Clausius-Clapeyron
kv estimate at the candidate wall temperature) alongside heat-flux
continuity; alpha_m in that solve also uses the stale kv[-1], same lag.
GRADIENTS NOT VALIDATED for masstrans=1 (with or without medtherm=1).

The main API is not valid for radial=6, arbitrary constitutive callbacks, PTT,
or Giesekus. The latter two are implemented only in the separate reduced solver.
Re-validate before extending any branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from thermal_fd import finite_diff_mat

P8 = 101325.0      # far-field pressure (Pa)
RHO = 1064.0       # far-field density (kg/m^3)
SURF = 0.07        # surface tension (N/m)
KAPPA = 1.4        # polytropic exponent (see module docstring)
C8 = 1484.0        # far-field sound speed (m/s)

# Tait equation-of-state constants for the liquid, IMRv2 defaults
# (default_case.m: GAM, nstate), used by radial=3,4.
_GAM_TAIT = 3049.13e5
_NSTATE_TAIT = 7.15

# Mie-Gruneisen EoS constants for radial=5,6, IMRv2 defaults (default_case.m:
# hugoniot_s; f_imr_fd.m: nog=(nstate-1)/2, same nstate as the Tait branch).
_HUGONIOT_S = 1.65
_NOG = (_NSTATE_TAIT - 1.0) / 2.0

# gas / vapor thermal-conductivity linear-in-T fit coefficients, IMRv2 defaults
# (default_case.m). K8 is IMRv2's reference conductivity: it mixes gas AND
# vapor coefficients even when vapor=0, because it is used purely as a
# normalization constant, not a physical mixture average at a given state.
_ATG, _BTG = 5.28e-5, 1.165e-2
_ATV, _BTV = 3.30e-5, 1.742e-2

# liquid (medium) thermal properties, IMRv2 defaults (default_case.m); water-like
_KM = 0.55          # liquid thermal conductivity (W/m/K)
_CP = 4.181e3        # liquid specific heat (J/kg/K)
_LT = 2.0            # exterior-grid stretching length (default_case.m)

# mass-transfer / vapor-species properties, IMRv2 defaults (default_case.m)
_RU = 8.3144598      # universal gas constant (J/mol/K)
_MWV = 18.01528e-3   # molar mass, water vapor (kg/mol)
_MWG = 28.966e-3     # molar mass, non-condensible gas / air (kg/mol)
_D0 = 24.2e-6        # binary (vapor-in-gas) diffusion coefficient (m^2/s)
_LHEAT = 2264.76e3   # latent heat of vaporization (J/kg)
_STRESS_STATE_COUNTS = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}


class SimulationError(RuntimeError):
    """Raised when the numerical integrator cannot complete a simulation."""

    def __init__(self, message: str, stats: SolverStats | None = None):
        super().__init__(message)
        self.stats = stats


@dataclass(frozen=True, slots=True)
class PhysicalParameters:
    """Dimensional environment and transport properties."""

    far_field_pressure_pa: float = P8
    medium_density_kg_m3: float = RHO
    surface_tension_n_m: float = SURF
    sound_speed_m_s: float = C8
    polytropic_exponent: float = KAPPA
    tait_pressure_pa: float = _GAM_TAIT
    tait_exponent: float = _NSTATE_TAIT
    hugoniot_slope: float = _HUGONIOT_S
    gas_conductivity_slope: float = _ATG
    gas_conductivity_offset: float = _BTG
    vapor_conductivity_slope: float = _ATV
    vapor_conductivity_offset: float = _BTV
    medium_conductivity_w_m_k: float = _KM
    medium_specific_heat_j_kg_k: float = _CP
    medium_grid_length: float = _LT
    mass_diffusivity_m2_s: float = _D0
    latent_heat_j_kg: float = _LHEAT
    universal_gas_constant_j_mol_k: float = _RU
    vapor_molar_mass_kg_mol: float = _MWV
    gas_molar_mass_kg_mol: float = _MWG

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"physics.{name} must be finite and positive")
        if self.polytropic_exponent <= 1.0:
            raise ValueError("physics.polytropic_exponent must be greater than 1")
        if self.tait_exponent <= 1.0:
            raise ValueError("physics.tait_exponent must be greater than 1")


@dataclass(frozen=True, slots=True)
class SampledForcing:
    """Far-field pressure perturbation sampled in dimensional units.

    Pressure is relative to the far-field baseline. A shape-preserving cubic
    is used between samples and the perturbation is zero outside their span.
    """

    time_s: tuple[float, ...]
    pressure_pa: tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.time_s)
        pressure = tuple(float(value) for value in self.pressure_pa)
        if len(times) < 2 or len(times) != len(pressure):
            raise ValueError("sampled forcing requires equal arrays of at least 2 values")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(pressure)):
            raise ValueError("sampled forcing values must be finite")
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("sampled forcing times must be non-negative and increasing")
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "pressure_pa", pressure)


@dataclass(frozen=True, slots=True)
class InitialState:
    """Optional dimensional initial conditions and internal solver state.

    ``stress_state`` uses the solver's nondimensional auxiliary variables.
    """

    wall_velocity_m_s: float = 0.0
    internal_pressure_pa: float | None = None
    bubble_temperature_k: float | None = None
    medium_temperature_k: float | None = None
    vapor_mass_fraction: float | None = None
    stress_state: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.wall_velocity_m_s):
            raise ValueError("initial.wall_velocity_m_s must be finite")
        for name in (
            "internal_pressure_pa",
            "bubble_temperature_k",
            "medium_temperature_k",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"initial.{name} must be finite and positive")
        fraction = self.vapor_mass_fraction
        if fraction is not None and (
            not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError("initial.vapor_mass_fraction must be between 0 and 1")
        if self.stress_state is not None:
            state = tuple(float(value) for value in self.stress_state)
            if not np.all(np.isfinite(state)):
                raise ValueError("initial.stress_state must contain finite values")
            object.__setattr__(self, "stress_state", state)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Validated dimensional inputs for one IMR simulation.

    The defaults match :func:`simulate`.
    """

    R0: float
    Req: float
    G: float
    mu: float
    stress: int = 1
    lam1: float = 0.0
    lam2: float = 0.0
    alphax: float = 0.25
    radial: int = 1
    vapor: int = 0
    T8: float = 298.15
    pA: float = 0.0
    omega: float = 0.0
    TW: float = 0.0
    DT: float = 0.0
    mn: float = 0.0
    wave_type: int = 0
    bubtherm: int = 0
    Nt: int = 25
    medtherm: int = 0
    Mt: int = 25
    masstrans: int = 0
    rtol: float = 1e-8
    atol: float = 1e-10
    physics: PhysicalParameters = field(default_factory=PhysicalParameters)
    sampled_forcing: SampledForcing | None = None
    initial: InitialState = field(default_factory=InitialState)

    def __post_init__(self) -> None:
        if not isinstance(self.physics, PhysicalParameters):
            raise TypeError("physics must be PhysicalParameters")
        if not isinstance(self.initial, InitialState):
            raise TypeError("initial must be InitialState")
        if self.sampled_forcing is not None and not isinstance(
            self.sampled_forcing, SampledForcing
        ):
            raise TypeError("sampled_forcing must be SampledForcing")
        _validate_inputs(
            [0.0, 1.0], self.R0, self.Req, self.G, self.mu, self.stress,
            self.lam1, self.lam2, self.alphax, self.radial, self.vapor,
            self.T8, self.pA, self.omega, self.TW, self.DT, self.mn,
            self.wave_type, self.bubtherm, self.Nt, self.medtherm, self.Mt,
            self.masstrans, self.rtol, self.atol,
        )
        if self.sampled_forcing is not None and (
            self.pA != 0.0
            or self.omega != 0.0
            or self.TW != 0.0
            or self.DT != 0.0
            or self.mn != 0.0
            or self.wave_type != 0
        ):
            raise ValueError("sampled_forcing cannot be combined with analytic forcing")


@dataclass(frozen=True, slots=True)
class SolverStats:
    """Integrator diagnostics for one completed or failed solve."""

    backend: str
    success: bool
    message: str
    nfev: int
    njev: int
    nlu: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Named locations of fields in the nondimensional ODE state."""

    pressure: int | None
    bubble_thermal: slice | None
    medium_thermal: slice | None
    vapor_fraction: slice | None
    stress: slice
    size: int

    @classmethod
    def from_config(cls, config: SimulationConfig) -> StateLayout:
        cursor = 2
        pressure = None
        bubble_thermal = None
        medium_thermal = None
        vapor_fraction = None
        if config.bubtherm:
            pressure = cursor
            cursor += 1
            bubble_thermal = slice(cursor, cursor + config.Nt)
            cursor += config.Nt
            if config.medtherm:
                medium_thermal = slice(cursor, cursor + config.Mt)
                cursor += config.Mt
            if config.masstrans:
                vapor_fraction = slice(cursor, cursor + config.Nt)
                cursor += config.Nt
        stress = slice(cursor, cursor + _STRESS_STATE_COUNTS[config.stress])
        cursor = stress.stop
        return cls(
            pressure=pressure,
            bubble_thermal=bubble_thermal,
            medium_thermal=medium_thermal,
            vapor_fraction=vapor_fraction,
            stress=stress,
            size=cursor,
        )


@dataclass(frozen=True, slots=True)
class MediumOperators:
    """Prepared geometry and finite-difference operators for the liquid layer."""

    xi: np.ndarray
    yT: np.ndarray
    yT2: np.ndarray
    yT3: np.ndarray
    iyT3: np.ndarray
    iyT4: np.ndarray
    iyT6: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    grad_Tm: np.ndarray
    grad_Trans: np.ndarray
    grad_C: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedForcing:
    """Nondimensional piecewise-cubic forcing coefficients."""

    knots: np.ndarray
    coefficients: np.ndarray


@dataclass(slots=True)
class _WallState:
    """Mutable wall-solve continuation state, private to one integration."""

    theta: float = -1e-4


@dataclass(frozen=True, slots=True)
class PreparedProblem:
    """Reusable static data for repeated simulations of one configuration."""

    config: SimulationConfig
    parameters: Mapping[str, float]
    layout: StateLayout
    initial_state: np.ndarray
    bubble_grid: np.ndarray | None = None
    bubble_D1: np.ndarray | None = None
    bubble_D2: np.ndarray | None = None
    medium: MediumOperators | None = None
    forcing: PreparedForcing | None = None

    def solve(self, tv) -> SimulationResult:
        """Run the prepared problem and return full physical outputs."""
        return _solve_prepared(self, tv)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable physical histories returned by the strict public API."""

    time_s: np.ndarray
    radius_ratio: np.ndarray
    wall_velocity_m_s: np.ndarray
    internal_pressure_pa: np.ndarray
    stress_integral_pa: np.ndarray
    bubble_temperature_k: np.ndarray | None
    medium_temperature_k: np.ndarray | None
    vapor_mass_fraction: np.ndarray | None
    stress_state: np.ndarray | None
    stats: SolverStats
    config: SimulationConfig

    @property
    def radius_m(self) -> np.ndarray:
        """Dimensional bubble radius in metres."""
        radius = self.config.R0 * self.radius_ratio
        radius.setflags(write=False)
        return radius


def _readonly_float_array(values) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _readonly_optional(values) -> np.ndarray | None:
    return None if values is None else _readonly_float_array(values)


def _validate_inputs(tv, R0, Req, G, mu, stress, lam1, lam2, alphax, radial,
                     vapor, T8, pA, omega, TW, DT, mn, wave_type, bubtherm,
                     Nt, medtherm, Mt, masstrans, rtol, atol) -> np.ndarray:
    """Validate the public API before constructing nondimensional groups."""
    times = np.asarray(tv, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("tv must be a one-dimensional array with at least two times")
    if not np.all(np.isfinite(times)):
        raise ValueError("tv must contain only finite values")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("tv must be non-negative and strictly increasing")

    for name, value in (("R0", R0), ("Req", Req), ("mu", mu), ("T8", T8),
                        ("rtol", rtol), ("atol", atol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(G) or G < 0:
        raise ValueError("G must be finite and non-negative")
    for name, value in (("lam1", lam1), ("lam2", lam2), ("alphax", alphax),
                        ("pA", pA), ("omega", omega), ("TW", TW), ("DT", DT),
                        ("mn", mn)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if lam1 < 0 or lam2 < 0:
        raise ValueError("lam1 and lam2 must be non-negative")
    if stress in (3, 4, 5) and lam1 <= 0:
        raise ValueError(f"stress={stress} requires lam1 > 0")
    if lam1 == 0 and lam2 != 0:
        raise ValueError("lam2 must be zero when lam1 is zero")

    for name, value, allowed in (
        ("stress", stress, range(0, 6)),
        ("radial", radial, range(1, 6)),
        ("wave_type", wave_type, range(0, 4)),
    ):
        if not isinstance(value, Integral) or value not in allowed:
            choices = ", ".join(str(choice) for choice in allowed)
            raise ValueError(f"{name} must be one of: {choices}")
    for name, value in (("vapor", vapor), ("bubtherm", bubtherm),
                        ("medtherm", medtherm), ("masstrans", masstrans)):
        if not isinstance(value, Integral) or value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
    for name, value in (("Nt", Nt), ("Mt", Mt)):
        if not isinstance(value, Integral) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")

    if medtherm and not bubtherm:
        raise ValueError("medtherm=1 requires bubtherm=1")
    if masstrans and not bubtherm:
        raise ValueError("masstrans=1 requires bubtherm=1")
    if masstrans and not vapor:
        raise ValueError("masstrans=1 requires vapor=1")
    if bubtherm and vapor and not masstrans:
        raise ValueError("bubtherm=1 with vapor=1 currently requires masstrans=1")
    return times


def pvsat(T):
    """Saturation vapour pressure (Pa), f_pvsat.m."""
    return 1.17e11 * np.exp(-5200.0 / T)


def params(R0, Req, G, mu, lam1=0.0, lam2=0.0, alphax=0.25, vapor=0, T8=298.15,
           pA=0.0, omega=0.0, TW=0.0, DT=0.0, mn=0.0, wave_type=0, bubtherm=0,
           masstrans=0, physics=None):
    """Nondimensional groups, matching f_call_params.m.

    P0's exponent depends on bubtherm (f_call_params.m:158-163): 3*kappa
    (polytropic/adiabatic initial-state assumption) when bubtherm==0, but
    plain 3 (pure geometric volume relation) when bubtherm!=0 -- these are
    NOT the same formula, and using the wrong one silently gives a P(0)
    off by ~8.6x for these test parameters, not a subtle error.
    """
    physics = PhysicalParameters() if physics is None else physics
    P8_value = physics.far_field_pressure_pa
    density = physics.medium_density_kg_m3
    surface_tension = physics.surface_tension_n_m
    kappa = physics.polytropic_exponent
    Uc = np.sqrt(P8_value / density)
    t0 = R0 / Uc
    Ca = P8_value / G if G > 0 else np.inf
    Re8 = P8_value * R0 / (mu * Uc)
    We = P8_value * R0 / (2 * surface_tension)
    Pv = vapor * pvsat(T8)
    P0_exp = 3 if bubtherm else 3 * kappa
    P0 = (
        P8_value + 2 * surface_tension / Req - Pv
    ) * (Req / R0) ** P0_exp
    Pv_star = Pv / P8_value
    Pb = P0 / P8_value + Pv_star
    # thermal groups (f_call_params.m); cheap, computed unconditionally so
    # bubtherm=1 needs no separate params() path
    K8 = 0.5 * (
        physics.gas_conductivity_slope * T8
        + physics.gas_conductivity_offset
        + physics.vapor_conductivity_slope * T8
        + physics.vapor_conductivity_offset
    )
    chi = T8 * K8 / (P8_value * R0 * Uc)
    alpha_g = physics.gas_conductivity_slope * T8 / K8
    beta_g = physics.gas_conductivity_offset / K8
    alpha_v = physics.vapor_conductivity_slope * T8 / K8
    beta_v = physics.vapor_conductivity_offset / K8
    # medium (liquid) thermal groups, f_call_params.m -- Dm=Km/(rho*Cp) is the
    # standard liquid thermal-diffusivity formula (confirmed from source, not
    # assumed); needed only when medtherm=1 but cheap to compute unconditionally
    Dm = (
        physics.medium_conductivity_w_m_k
        / (density * physics.medium_specific_heat_j_kg_k)
    )
    Foh = Dm / (Uc * R0)
    iota = (
        physics.medium_conductivity_w_m_k
        / (K8 * physics.medium_grid_length)
    )
    Br = Uc ** 2 / (physics.medium_specific_heat_j_kg_k * T8)
    # mass-transfer / vapor-species groups (f_call_params.m). kv0 (the initial/
    # reference vapor mass fraction) is only physically meaningful for vapor=1
    # (Pv_star>0) -- confirmed from source: with Pv_star=0 the mass-ratio
    # formula below is a 0/0-type limit that IEEE arithmetic resolves to
    # kv0=0, matching the dry-gas (bubtherm-only) case exactly.
    Rv = (
        physics.universal_gas_constant_j_mol_k
        / physics.vapor_molar_mass_kg_mol
    )
    Rg = (
        physics.universal_gas_constant_j_mol_k
        / physics.gas_molar_mass_kg_mol
    )
    Rnondim = P8_value / (density * T8)
    Rv_star = Rv / Rnondim
    Rg_star = Rg / Rnondim
    Fom = physics.mass_diffusivity_m2_s / (Uc * R0)
    L_heat_star = physics.latent_heat_j_kg / Uc ** 2
    if masstrans and vapor == 0:
        raise ValueError("masstrans=1 requires vapor=1 (kv0's formula is only "
                          "physically meaningful with Pv_star>0)")
    if Pv_star > 0:
        kv0 = 1.0 / (1.0 + (Rv_star / Rg_star) * (Pb / Pv_star - 1.0))
    else:
        kv0 = 0.0
    tait_gamma = physics.tait_pressure_pa / P8_value
    tait_sam = 1.0 + tait_gamma
    tait_no = (physics.tait_exponent - 1.0) / physics.tait_exponent
    Cstar = physics.sound_speed_m_s / Uc
    nog = (physics.tait_exponent - 1.0) / 2.0
    mie_reference = _mie_F(
        _mu_of_A(1.0 / Cstar ** 2, physics.hugoniot_slope, nog),
        physics.hugoniot_slope,
        nog,
    )
    return dict(t0=t0, Uc=Uc, P8=P8_value, Ca=Ca, Re8=Re8,
                iWe=1.0 / We, req=Req / R0, Pb=Pb, Pv=Pv_star, T8=T8,
                kappa=kappa, kapover=(kappa - 1.0) / kappa,
                De=(lam1 * Uc / R0 if lam1 > 0 else 0.0),
                LAM=(lam2 / lam1 if lam1 > 0 else 0.0),
                Cstar=Cstar, alphax=alphax,
                tait_gamma=tait_gamma, tait_sam=tait_sam, tait_no=tait_no,
                tait_exponent=physics.tait_exponent,
                hugoniot_slope=physics.hugoniot_slope, nog=nog,
                mie_reference=mie_reference,
                ee=pA / P8_value, om=omega * t0, tw=TW / t0, dt=DT / t0, mn=mn,
                wave_type=wave_type, chi=chi, alpha_g=alpha_g, beta_g=beta_g,
                alpha_v=alpha_v, beta_v=beta_v,
                Foh=Foh, iota=iota, Br=Br, Lt=physics.medium_grid_length,
                Rv_star=Rv_star, Rg_star=Rg_star, Fom=Fom,
                L_heat_star=L_heat_star, kv0=kv0)


def _stress(stress, p, R, Rd, Z):
    """Return (S, Sdot, dZdt) matching f_stress.m."""
    Rst = p['req'] / R
    Ca, Re8, De, LAM, ax = p['Ca'], p['Re8'], p['De'], p['LAM'], p['alphax']
    if stress == 0:                                     # no stress
        return 0.0, 0.0, None
    if stress == 1:                                     # neo-Hookean KV
        S = -(5 - 4 * Rst - Rst ** 4) / (2 * Ca) - 4.0 / Re8 * Rd / R
        Sdot = -2 * Rd / R * (Rst + Rst ** 4) / Ca + 4.0 / Re8 * (Rd / R) ** 2
        return S, Sdot, None
    if stress == 2:                                     # quadratic KV
        S = ((3 * ax - 1) * (5 - Rst ** 4 - 4 * Rst) / (2 * Ca)
             - 4.0 / Re8 * Rd / R
             + (2 * ax / Ca) * (27 / 40 + Rst ** 8 / 8 + Rst ** 5 / 5 + Rst ** 2 - 2 / Rst))
        Sdot = ((Rd / R) * ((3 * ax - 1) / (2 * Ca)) * (4 * Rst ** 4 + 4 * Rst)
                + 4 * (Rd / R) ** 2 / Re8
                - 2 * ax / Ca * Rd / R * (Rst ** 8 + Rst ** 5 + 2 * Rst ** 2 + 2 / Rst))
        return S, Sdot, None
    if stress == 3:                                     # Zener / Jeffreys / Maxwell
        Z1 = Z[0]
        S = Z1 / R ** 3 - 4 * LAM / Re8 * Rd / R
        Ze = -0.5 * (R ** 3 / Ca) * (5 - Rst ** 4 - 4 * Rst)
        Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        Sdot = Z1d / R ** 3 - 3 * Rd / R ** 4 * Z1 + 4 * LAM / Re8 * (Rd / R) ** 2
        return S, Sdot, np.array([Z1d])
    if stress == 4:                                     # Zener / Jeffreys / Maxwell, quadratic KV
        Z1 = Z[0]
        S = Z1 / R ** 3 - 4 * LAM / Re8 * Rd / R
        strainhard = (3 * ax - 1) / (2 * Ca)
        Ze = R ** 3 * (strainhard * (5 - Rst ** 4 - 4 * Rst)
                       + (2 * ax / Ca) * (0.675 + 0.125 * Rst ** 8
                                          + 0.2 * Rst ** 5 + Rst ** 2 - 2 / Rst))
        Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        Sdot = Z1d / R ** 3 - 3 * Rd / R ** 4 * Z1 + 4 * LAM / Re8 * Rd ** 2 / R ** 2
        return S, Sdot, np.array([Z1d])
    if stress == 5:                                     # UCM / Oldroyd-B
        Z1, Z2 = Z[0], Z[1]
        Z1d = -(1 / De - 2 * Rd / R) * Z1 + 2 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        Z2d = -(1 / De + Rd / R) * Z2 + 2 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        S = (Z1 + Z2) / R ** 3 - 4 * LAM / Re8 * Rd / R
        Sdot = ((Z1d + Z2d) / R ** 3 - 3 * Rd / R ** 4 * (Z1 + Z2)
                + 4 * LAM / Re8 * Rd ** 2 / R ** 2)
        return S, Sdot, np.array([Z1d, Z2d])
    raise ValueError(f"stress={stress} not supported")


def _sampled_pressure(tn, forcing):
    if tn < forcing.knots[0] or tn > forcing.knots[-1]:
        return 0.0, 0.0
    interval = np.searchsorted(forcing.knots, tn, side="right") - 1
    interval = min(interval, forcing.knots.size - 2)
    offset = tn - forcing.knots[interval]
    c0, c1, c2, c3 = forcing.coefficients[:, interval]
    pressure = ((c0 * offset + c1) * offset + c2) * offset + c3
    pressure_rate = (3.0 * c0 * offset + 2.0 * c1) * offset + c2
    return pressure, pressure_rate


def _pinf(tn, p, forcing=None):
    """Far-field pressure perturbation (Pf8, Pf8dot), f_pinfinity.m. Nondimensional."""
    if forcing is not None:
        return _sampled_pressure(tn, forcing)
    wt, ee, om, tw, dt, mn = (p['wave_type'], p['ee'], p['om'], p['tw'], p['dt'], p['mn'])
    if ee == 0.0:
        return 0.0, 0.0
    if wt == 0:                                        # constant offset impulse
        return ee, 0.0
    if wt == 1:                                        # Gaussian
        e = np.exp(-((tn - dt) ** 2) / tw ** 2)
        return -ee * e, ee * (2 * (tn - dt) / tw ** 2) * e
    if wt == 2:                                        # histotripsy pulse
        if tn < dt - np.pi / om or tn > dt + np.pi / om:
            return 0.0, 0.0
        c = 0.5 + 0.5 * np.cos(om * (tn - dt))
        return (ee * c ** mn,
                -ee * mn * c ** (mn - 1) * 0.5 * om * np.sin(om * (tn - dt)))
    if wt == 3:                                        # Heaviside step
        return (-ee * (1.0 - (1.0 if tn > tw else 0.0)), 0.0)
    raise ValueError(f"wave_type={wt} not supported")


def _nZ(stress):
    """Number of internal stress variables."""
    return _STRESS_STATE_COUNTS[stress]


def _freeze_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _prepare_forcing(config, parameters):
    if config.sampled_forcing is None:
        return None
    forcing = config.sampled_forcing
    knots = np.asarray(forcing.time_s) / parameters["t0"]
    values = np.asarray(forcing.pressure_pa) / parameters["P8"]
    interpolant = PchipInterpolator(knots, values, extrapolate=False)
    return PreparedForcing(
        knots=_freeze_array(knots),
        coefficients=_freeze_array(interpolant.c),
    )


def _thermal_state(temperature_ratio, alpha):
    shifted = 1.0 + alpha * (temperature_ratio - 1.0)
    return (shifted ** 2 - 1.0) / (2.0 * alpha)


def prepare(config: SimulationConfig) -> PreparedProblem:
    """Prepare reusable grids, operators, parameters, and state layout."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    p = params(
        config.R0, config.Req, config.G, config.mu, config.lam1, config.lam2,
        config.alphax, config.vapor, config.T8, config.pA, config.omega,
        config.TW, config.DT, config.mn, config.wave_type, config.bubtherm,
        config.masstrans, config.physics,
    )
    initial = config.initial
    if initial.internal_pressure_pa is not None:
        p["Pb"] = initial.internal_pressure_pa / p["P8"]
    layout = StateLayout.from_config(config)
    initial_state = np.zeros(layout.size)
    initial_state[0] = 1.0
    initial_state[1] = initial.wall_velocity_m_s / p["Uc"]

    if initial.bubble_temperature_k is not None and not config.bubtherm:
        raise ValueError("initial bubble temperature requires bubtherm=1")
    if initial.medium_temperature_k is not None and not config.medtherm:
        raise ValueError("initial medium temperature requires medtherm=1")
    if initial.vapor_mass_fraction is not None and not config.masstrans:
        raise ValueError("initial vapor mass fraction requires masstrans=1")
    stress_width = layout.stress.stop - layout.stress.start
    if initial.stress_state is not None and len(initial.stress_state) != stress_width:
        raise ValueError(
            f"initial stress_state requires exactly {stress_width} values"
        )
    if initial.stress_state is not None:
        initial_state[layout.stress] = initial.stress_state

    bubble_grid = None
    bubble_D1 = None
    bubble_D2 = None
    medium = None
    if config.bubtherm:
        initial_state[layout.pressure] = p["Pb"]
        bubble_grid = _freeze_array(np.linspace(0.0, 1.0, config.Nt))
        bubble_D1 = _freeze_array(finite_diff_mat(config.Nt, 1, tm_check=0))
        bubble_D2 = _freeze_array(finite_diff_mat(config.Nt, 2, tm_check=0))
        vapor_fraction = (
            p["kv0"]
            if initial.vapor_mass_fraction is None
            else initial.vapor_mass_fraction
        )
        if config.masstrans:
            initial_state[layout.vapor_fraction] = vapor_fraction
        temperature_ratio = (
            1.0
            if initial.bubble_temperature_k is None
            else initial.bubble_temperature_k / config.T8
        )
        alpha = (
            vapor_fraction * p["alpha_v"]
            + (1.0 - vapor_fraction) * p["alpha_g"]
            if config.masstrans
            else p["alpha_g"]
        )
        initial_state[layout.bubble_thermal] = _thermal_state(
            temperature_ratio, alpha
        )
        if config.medtherm:
            medium_temperature_ratio = (
                1.0
                if initial.medium_temperature_k is None
                else initial.medium_temperature_k / config.T8
            )
            initial_state[layout.medium_thermal] = medium_temperature_ratio
            Nm = config.Mt - 1
            deltaYm = -2.0 / Nm
            xi = 1.0 + np.arange(config.Mt) * deltaYm
            with np.errstate(divide="ignore", invalid="ignore"):
                yT = (2.0 / (xi + 1.0) - 1.0) * p["Lt"] + 1.0
                yT2, yT3 = yT ** 2, yT ** 3
                iyT3, iyT4, iyT6 = yT ** -3, yT ** -4, yT ** -6
            coeff = np.array([-1.5, 2.0, -0.5])
            deltaY = 1.0 / (config.Nt - 1)
            medium = MediumOperators(
                xi=_freeze_array(xi),
                yT=_freeze_array(yT),
                yT2=_freeze_array(yT2),
                yT3=_freeze_array(yT3),
                iyT3=_freeze_array(iyT3),
                iyT4=_freeze_array(iyT4),
                iyT6=_freeze_array(iyT6),
                D1=_freeze_array(finite_diff_mat(config.Mt, 1, tm_check=1)),
                D2=_freeze_array(finite_diff_mat(config.Mt, 2, tm_check=1)),
                grad_Tm=_freeze_array(
                    2 * p["chi"] * p["iota"] / deltaYm * coeff
                ),
                grad_Trans=_freeze_array(-coeff * p["chi"] / deltaY),
                grad_C=_freeze_array(
                    -coeff * p["Fom"] * p["L_heat_star"] / deltaY
                ),
            )
    return PreparedProblem(
        config=config,
        parameters=MappingProxyType(p),
        layout=layout,
        initial_state=_freeze_array(initial_state),
        bubble_grid=bubble_grid,
        bubble_D1=bubble_D1,
        bubble_D2=bubble_D2,
        medium=medium,
        forcing=_prepare_forcing(config, p),
    )


def _JdotA(stress, p):
    """f_call_params.m: 4/Re8 for stress 1-4, 4*LAM/Re8 for stress 5-6."""
    return 4.0 / p['Re8'] if stress in (1, 2, 3, 4) else 4.0 * p['LAM'] / p['Re8']


def _mu_of_A(A, s=_HUGONIOT_S, nog=_NOG):
    """Compression mu solving a*mu^2+b*mu+A=0 (f_mie_gruneisen_eos_scalar.m)."""
    a = A * s ** 2 - nog
    b = -2.0 * A * s - 1.0
    d = b ** 2 - 4.0 * a * A
    return (-b + np.sqrt(d)) / (2.0 * a)


def _mie_F(mu, s=_HUGONIOT_S, nog=_NOG):
    """Closed-form antiderivative of rho(mu)*dA/dmu, i.e. of the integrand
    MATLAB evaluates via numerical quadrature (integral(...)) every RHS call
    to get hB. Derived via partial fractions and verified against
    scipy.integrate.quad to ~1e-12 over a wide pressure range; using this
    instead of live quadrature is what keeps radial=5,6 fast."""
    w = 1.0 - s * mu
    return ((2 * nog + s - 1) / (s + 1) ** 3 * np.log(w / (mu + 1.0))
            + (nog + s) / (s * (s + 1) * w ** 2)
            - (2 * nog + s - 1) / ((s + 1) ** 2 * w))


def _mie_gruneisen(P, Cstar, s, nog, reference):
    """(C, hB, hH) at nondimensional pressure P, f_mie_gruneisen_eos_scalar.m.
    hB = integral_1^P rho(p) dp via the closed form above, not live quadrature."""
    A = P / Cstar ** 2
    mu = _mu_of_A(A, s, nog)
    w = 1.0 - s * mu
    # transient negative discriminant on rejected LSODA trial steps only --
    # the accepted trajectory stays real (confirmed against real IMRv2, see
    # module docstring); harmless, same pattern as the xi[-1]=-1 case below.
    with np.errstate(invalid='ignore'):
        C = Cstar * np.sqrt(((1 + 2 * nog * mu) * w ** 2 + 2 * s * mu * (1 + nog * mu) * w) / w ** 4)
    hH = 1.0 / (1.0 + mu)
    hB = Cstar ** 2 * (_mie_F(mu, s, nog) - reference)
    return C, hB, hH


def _dissipation(stress, p, R, Rd, yT2, yT3, iyT3, iyT4, iyT6):
    """taugradu, f_stress_dissipation.m, finite-difference (non-spectral) mode.
    fnu=0 always (nu_model not supported), so Br/(Re8+DRe*fnu) simplifies to
    Br/Re8 exactly."""
    Ca, Re8, Br, ax = p['Ca'], p['Re8'], p['Br'], p['alphax']
    Rst = p['req'] / R
    x2 = (yT3 - 1.0 + Rst ** 3) ** (2.0 / 3.0)
    ix2 = 1.0 / x2
    x4 = x2 ** 2
    base = (12.0 * (Br / Re8) * (Rd / R) ** 2 * iyT6
            + 2.0 * Br / Ca * iyT3 * (Rd / R) * (yT2 * ix2 - iyT4 * x4))
    if stress == 2:
        return base * (1.0 + ax * (x4 * iyT4 + 2.0 * yT2 * ix2 - 3.0))
    return base   # stress in {1, 3, 5}: identical to the base formula


def _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale):
    """Wall vapor mass fraction from vapor-liquid equilibrium, f_kv_of_T.m.
    Tw is nondimensional; pvsat() takes a dimensional temperature."""
    theta_var = Rvg_ratio * (P / (pvsat(Tw * T8) / pressure_scale) - 1.0)
    return 1.0 / (1.0 + theta_var)


def _secant_root(function, guess, *, tol=1e-13, maxiter=100):
    """Scalar secant solve without SciPy's array-oriented dispatch overhead."""
    p0 = float(guess)
    p1 = p0 * 1.0001
    p1 += 1e-4 if p1 >= 0.0 else -1e-4
    q0 = function(p0)
    q1 = function(p1)
    if abs(q1) < abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0

    for _ in range(maxiter):
        if q1 == q0:
            raise RuntimeError("wall boundary secant solve encountered zero slope")
        if abs(q1) > abs(q0):
            ratio = q0 / q1
            root = (-ratio * p1 + p0) / (1.0 - ratio)
        else:
            ratio = q1 / q0
            root = (-ratio * p0 + p1) / (1.0 - ratio)
        if abs(root - p1) <= tol:
            return root
        p0, q0 = p1, q1
        p1 = root
        q1 = function(p1)
    raise RuntimeError(
        f"wall boundary secant solve failed to converge after {maxiter} iterations"
    )


def _wall_theta_bw(guess, theta_tail, Tm_tail, alpha_g, grad_Tm, grad_Trans):
    """Solve heat-flux continuity at the wall for theta[-1], f_bubble_wall_thermal_bc.m.
    theta_tail = [theta[-2], theta[-3]], Tm_tail = [Tm[1], Tm[2]] (0-indexed)."""
    def resid(theta_bw):
        Tw = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_g)) / alpha_g
        lhs = grad_Tm[0] * Tw + grad_Tm[1] * Tm_tail[0] + grad_Tm[2] * Tm_tail[1]
        rhs = grad_Trans[0] * theta_bw + grad_Trans[1] * theta_tail[0] + grad_Trans[2] * theta_tail[1]
        return lhs + rhs
    return _secant_root(resid, guess)


def _wall_theta_bw_full(guess, theta_tail, Tm_tail, kv_tail, kv_end_stale, P,
                         alpha_v, alpha_g, T8, Rvg_ratio, Rva_diff, Rg_star,
                         pressure_scale, grad_Tm, grad_Trans, grad_C):
    """Solve coupled heat-flux + vapor-mass-flux continuity at the wall for
    theta[-1], f_bubble_wall_full_bc.m. alpha_m uses kv_end_stale (the raw,
    pre-update kv[-1] state) -- the same one-step lag as _kv_of_T's own
    caller, not reconciled/updated mid-solve. theta_tail/Tm_tail/kv_tail =
    [x[-2], x[-3]] (0-indexed), matching MATLAB's end-1:-1:end-2."""
    alpha_m = kv_end_stale * alpha_v + (1.0 - kv_end_stale) * alpha_g

    def resid(theta_bw):
        Tw = (alpha_m - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_m)) / alpha_m
        kvw = _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale)
        lhs = grad_Tm[0] * Tw + grad_Tm[1] * Tm_tail[0] + grad_Tm[2] * Tm_tail[1]
        rhs = grad_Trans[0] * theta_bw + grad_Trans[1] * theta_tail[0] + grad_Trans[2] * theta_tail[1]
        scalar = P / ((kvw * Rva_diff + Rg_star) * (Tw * (1.0 - kvw)))
        extra = scalar * (grad_C[0] * kvw + grad_C[1] * kv_tail[0] + grad_C[2] * kv_tail[1])
        return lhs + rhs + extra
    failure = None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            return _secant_root(resid, guess)
        except RuntimeError as error:
            failure = error
            roots = []
            for fallback in (0.0, 0.5 * guess, 1.5 * guess, theta_tail[0]):
                try:
                    roots.append(_secant_root(resid, fallback))
                except RuntimeError:
                    pass
    if roots:
        return min(roots, key=lambda root: abs(root - guess))
    raise failure


def _apply_thermal_boundaries(
    theta, Tm, kv, P, p, medium, masstrans, wall_state
):
    """Apply algebraic wall conditions to local state copies."""
    if medium is not None and masstrans:
        theta[-1] = _wall_theta_bw_full(
            wall_state.theta,
            [theta[-2], theta[-3]],
            [Tm[1], Tm[2]],
            [kv[-2], kv[-3]],
            kv[-1],
            P,
            p["alpha_v"],
            p["alpha_g"],
            p["T8"],
            p["Rv_star"] / p["Rg_star"],
            p["Rv_star"] - p["Rg_star"],
            p["Rg_star"],
            p["P8"],
            medium.grad_Tm,
            medium.grad_Trans,
            medium.grad_C,
        )
        wall_state.theta = theta[-1]
    elif medium is not None:
        theta[-1] = _wall_theta_bw(
            wall_state.theta,
            [theta[-2], theta[-3]],
            [Tm[1], Tm[2]],
            p["alpha_g"],
            medium.grad_Tm,
            medium.grad_Trans,
        )
        wall_state.theta = theta[-1]

    alpha_m = None
    if masstrans:
        alpha_m = kv * p["alpha_v"] + (1.0 - kv) * p["alpha_g"]
        temperature = (
            alpha_m - 1.0 + np.sqrt(1.0 + 2.0 * theta * alpha_m)
        ) / alpha_m
        kv[-1] = _kv_of_T(
            temperature[-1],
            P,
            p["T8"],
            p["Rv_star"] / p["Rg_star"],
            p["P8"],
        )
    else:
        alpha_g = p["alpha_g"]
        temperature = (
            alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta * alpha_g)
        ) / alpha_g
    if Tm is not None:
        Tm[0] = temperature[-1]
    return temperature, alpha_m


def _rhs(tn, y, p, stress, radial, bubtherm=0, D1=None, D2=None, ygrid=None,
          medtherm=0, mt=None, masstrans=0, wall_state=None, forcing=None):
    R = max(y[0], 1e-8)
    Rd = y[1]
    Pv = p['Pv']
    kappa = p["kappa"]

    kv = None
    if bubtherm:
        P = y[2]
        Nt = ygrid.size
        theta = y[3:3 + Nt].copy()
        idx = 3 + Nt
        Tm = None
        if medtherm:
            Tm = y[idx:idx + mt.xi.size].copy()
            idx += mt.xi.size
        if masstrans:
            kv = y[idx:idx + Nt].copy()
            idx += Nt
        T, alpha_m = _apply_thermal_boundaries(
            theta, Tm, kv, P, p, mt, masstrans, wall_state
        )
        Zstart = idx
    else:
        P = (p['Pb'] - Pv) * R ** (-3 * kappa) + Pv          # f_imr_fd.m:412
        Zstart = 2

    nz = _nZ(stress)
    Z = y[Zstart:Zstart + nz] if nz else None
    S, Sdot, dZ = _stress(stress, p, R, Rd, Z)
    Pf8, Pf8dot = _pinf(tn, p, forcing)
    iWe = p['iWe']

    thetadot = None
    kvdot = None
    if bubtherm:
        alpha_g, beta_g, chi = p['alpha_g'], p['beta_g'], p['chi']
        if masstrans:
            # f_imr_fd.m, "if bubtherm && masstrans" branch. T is computed with
            # the STALE (pre-update) kv[-1] -- matches source order exactly
            # (T computed once, THEN kv[-1] is freshly overwritten below); this
            # one-step lag is IMRv2's own behavior, not reconciled/"fixed" here.
            alpha_v, beta_v = p['alpha_v'], p['beta_v']
            Rv_star, Rg_star = p['Rv_star'], p['Rg_star']
            Rva_diff = Rv_star - Rg_star
            Fom = p['Fom']
            dtheta = D1 @ theta
            ddtheta = D2 @ theta
            dkv = D1 @ kv
            ddkv = D2 @ kv
            Rmix = kv * Rv_star + (1.0 - kv) * Rg_star
            RDkv = (Rva_diff / Rmix) * dkv
            Pdot = 3.0 / R * (chi * (kappa - 1.0) * dtheta[-1] / R - kappa * P * Rd
                              + kappa * P * Fom * Rv_star * dkv[-1]
                              / (T[-1] * R * Rmix[-1] * (1.0 - kv[-1])))
            Uvel = ((chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (kappa * P)
                    + Fom / R * RDkv)
            Kstar_g = alpha_g * T + beta_g
            Kstar_v = alpha_v * T + beta_v
            Kstar = kv * Kstar_v + (1.0 - kv) * Kstar_g
            nonlinear_term = (
                (chi * ddtheta / R ** 2 + Pdot)
                * (p["kapover"] * Kstar * T / P)
            )
            advection_term = -dtheta * (Uvel - ygrid * Rd) / R
            mass_diffusion = (Fom / R ** 2) * (Rva_diff / Rmix) * dkv * dtheta
            thetadot = advection_term + nonlinear_term + mass_diffusion
            thetadot[-1] = 0.0
            nonlinear_diffusion = dkv * (dtheta / (np.sqrt(1.0 + 2.0 * alpha_m * theta) * T) + RDkv)
            advection_term2 = (Uvel - Rd * ygrid) / R * dkv
            kvdot = Fom / R ** 2 * (ddkv - nonlinear_diffusion) - advection_term2
            kvdot[-1] = 0.0
        else:
            # f_imr_fd.m, "elseif bubtherm" branch, kv0=0 (dry gas) simplification.
            # Identical whether medtherm is on or off -- only theta[-1]'s VALUE
            # differs (wall-BC solve above vs. frozen at its initial value).
            dtheta = D1 @ theta
            ddtheta = D2 @ theta
            Pdot = 3.0 / R * (
                chi * (kappa - 1.0) * dtheta[-1] / R - kappa * P * Rd
            )
            Uvel = (
                chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0
            ) / (kappa * P)
            Kstar = alpha_g * T + beta_g
            diffusion = (
                (chi * ddtheta / R ** 2 + Pdot)
                * (p["kapover"] * Kstar * T / P)
            )
            advection = -dtheta * (Uvel - ygrid * Rd) / R
            thetadot = advection + diffusion
            thetadot[-1] = 0.0
    else:
        Pdot = -3 * kappa * (p['Pb'] - Pv) * R ** (-3 * kappa - 1) * Rd

    Tmdot = None
    if medtherm:
        # f_imr_fd.m "surrounding temperature" block. xi[-1]=-1 exactly (the
        # far-field point) makes yT[-1]=inf -- an algebraic singularity
        # inherent to IMRv2's own xi=1+(j-1)*deltaYm formula (confirmed:
        # xi_last = 1+Nm*(-2/Nm) = -1 identically, not introduced by this
        # port), producing a transient inf*0=nan in the last entry of
        # med_advection/taugradu. Harmless: Tmdot[-1]=0.0 unconditionally
        # overwrites it and that state slot's value never depends on it
        # (same frozen-slot pattern as theta[-1] without medtherm).
        dTm = mt.D1 @ Tm
        ddTm = mt.D2 @ Tm
        xi, yT, yT2, yT3, iyT3, iyT4, iyT6 = (
            mt.xi, mt.yT, mt.yT2, mt.yT3, mt.iyT3, mt.iyT4, mt.iyT6
        )
        Lt, Foh, iota = p['Lt'], p['Foh'], p['iota']
        with np.errstate(divide='ignore', invalid='ignore'):
            med_advection = ((1 + xi) ** 2 / (Lt * R)
                              * (Rd / yT2 * (1 - yT3) / 2 + Foh / R * ((xi + 1) / (2 * Lt) - 1 / yT))
                              * dTm)
            med_diffusion = Foh / R ** 2 * (xi + 1) ** 4 / Lt ** 2 * ddTm / 4
            taugradu = _dissipation(stress, p, R, Rd, yT2, yT3, iyT3, iyT4, iyT6)
        Tmdot = med_advection + med_diffusion + taugradu
        Tmdot[0] = 0.0
        Tmdot[-1] = 0.0

    if radial == 1:                                     # Rayleigh-Plesset
        Rdd = (P - 1 - Pf8 - iWe / R + S - 1.5 * Rd ** 2) / R
    elif radial == 2:                                   # Keller-Miksis (pressure form)
        Cs = p['Cstar']
        num = ((1 + Rd / Cs) * (P - 1 - Pf8 - iWe / R + S)
               + R / Cs * (Pdot + iWe * Rd / R ** 2 + Sdot - Pf8dot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) / Cs
        Rdd = num / den
    elif radial == 3:                                   # Keller-Miksis, enthalpy, Tait EoS
        Cs = p['Cstar']
        Pb = P - iWe / R + p["tait_gamma"] + S
        hB = (
            p["tait_sam"] / p["tait_no"]
            * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
        )
        hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
        num = ((1 + Rd / Cs) * (hB - Pf8) - R / Cs * Pf8dot
               + R / Cs * hH * (Pdot + iWe * Rd / R ** 2 + Sdot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) * hH / Cs
        Rdd = num / den
    elif radial == 4:                                   # Gilmore, Tait EoS
        Pb = P - iWe / R + p["tait_gamma"] + S
        rho = (Pb / p["tait_sam"]) ** (1.0 / p["tait_exponent"])
        Cs = np.sqrt(p["tait_exponent"] * Pb / rho)
        hB = (
            p["tait_sam"] / p["tait_no"]
            * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
        )
        hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
        num = ((1 + Rd / Cs) * (hB - Pf8) - R / Cs * Pf8dot
               + R / Cs * hH * (Pdot + iWe * Rd / R ** 2 + Sdot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) * hH / Cs
        Rdd = num / den
    elif radial == 5:                                   # Keller-Miksis, enthalpy, Mie-Gruneisen EoS
        # f_radial_eq.m: Pb = P - iWe/R (no +S here -- genuinely different
        # from radial=3/4's Pb, not reconciled/harmonized with them).
        Cs = p['Cstar']
        Pb = P - iWe / R
        _, hB, hH = _mie_gruneisen(
            Pb,
            Cs,
            p["hugoniot_slope"],
            p["nog"],
            p["mie_reference"],
        )
        num = ((1 + Rd / Cs) * (hB - Pf8) - R / Cs * Pf8dot
               + R / Cs * hH * (Pdot + iWe * Rd / R ** 2 + Sdot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) * hH / Cs
        Rdd = num / den
    else:
        # radial=6 (Gilmore, Mie-Gruneisen EoS) is NOT supported: as shipped,
        # f_radial_eq.m evaluates the EoS at raw P (no iWe/R correction, unlike
        # radial=5's Pb=P-iWe/R), which pushes the Gilmore sound-speed formula's
        # discriminant negative for essentially every polytropic-gas P tested
        # (confirmed against real IMRv2 -- it returns a COMPLEX R(t), not an
        # error, for everything from mild oscillations to deep collapse). This
        # is dead/broken code upstream, not a real reference to port against.
        raise ValueError(f"radial={radial} not supported")

    out = [Rd, Rdd]
    if bubtherm:
        out.append(Pdot)
        out.extend(thetadot.tolist())
    if medtherm:
        out.extend(Tmdot.tolist())
    if masstrans:
        out.extend(kvdot.tolist())
    if dZ is not None:
        out.extend(dZ.tolist())
    return out


def _radius_floor_event(_tn, y, *_args):
    return y[0] - 1e-8


_radius_floor_event.terminal = True
_radius_floor_event.direction = -1


def _thermal_outputs(problem: PreparedProblem, states: np.ndarray):
    config = problem.config
    if not config.bubtherm:
        return None, None, None
    layout = problem.layout
    p = problem.parameters
    count = states.shape[0]
    bubble_temperature = np.empty((count, config.Nt))
    medium_temperature = (
        np.empty((count, config.Mt)) if config.medtherm else None
    )
    vapor_fraction = (
        np.empty((count, config.Nt)) if config.masstrans else None
    )
    wall_state = _WallState()
    for index, state in enumerate(states):
        theta = state[layout.bubble_thermal].copy()
        Tm = (
            state[layout.medium_thermal].copy()
            if layout.medium_thermal is not None
            else None
        )
        kv = (
            state[layout.vapor_fraction].copy()
            if layout.vapor_fraction is not None
            else None
        )
        temperature, _ = _apply_thermal_boundaries(
            theta,
            Tm,
            kv,
            state[layout.pressure],
            p,
            problem.medium,
            config.masstrans,
            wall_state,
        )
        bubble_temperature[index] = config.T8 * temperature
        if medium_temperature is not None:
            medium_temperature[index] = config.T8 * Tm
        if vapor_fraction is not None:
            vapor_fraction[index] = kv
    return bubble_temperature, medium_temperature, vapor_fraction


def _build_result(
    problem: PreparedProblem,
    time_s: np.ndarray,
    solution,
    stats: SolverStats,
) -> SimulationResult:
    config = problem.config
    p = problem.parameters
    layout = problem.layout
    states = solution.y.T
    radius_ratio = states[:, 0]
    velocity = states[:, 1]
    Uc = p["Uc"]
    pressure_scale = p["P8"]
    kappa = p["kappa"]

    if layout.pressure is None:
        pressure = (
            (p["Pb"] - p["Pv"]) * radius_ratio ** (-3 * kappa) + p["Pv"]
        )
    else:
        pressure = states[:, layout.pressure]

    stress_integral = np.empty(states.shape[0])
    for index, state in enumerate(states):
        stress_state = (
            state[layout.stress] if layout.stress.stop > layout.stress.start else None
        )
        stress_integral[index] = _stress(
            config.stress, p, state[0], state[1], stress_state
        )[0]

    bubble_temperature, medium_temperature, vapor_fraction = _thermal_outputs(
        problem, states
    )
    internal_stress_state = (
        states[:, layout.stress]
        if layout.stress.stop > layout.stress.start
        else None
    )
    return SimulationResult(
        time_s=_readonly_float_array(time_s),
        radius_ratio=_readonly_float_array(radius_ratio),
        wall_velocity_m_s=_readonly_float_array(Uc * velocity),
        internal_pressure_pa=_readonly_float_array(pressure_scale * pressure),
        stress_integral_pa=_readonly_float_array(
            pressure_scale * stress_integral
        ),
        bubble_temperature_k=_readonly_optional(bubble_temperature),
        medium_temperature_k=_readonly_optional(medium_temperature),
        vapor_mass_fraction=_readonly_optional(vapor_fraction),
        stress_state=_readonly_optional(internal_stress_state),
        stats=stats,
        config=config,
    )


def _integrate_prepared(problem: PreparedProblem, tv):
    config = problem.config
    time_s = _validate_inputs(
        tv, config.R0, config.Req, config.G, config.mu, config.stress,
        config.lam1, config.lam2, config.alphax, config.radial, config.vapor,
        config.T8, config.pA, config.omega, config.TW, config.DT, config.mn,
        config.wave_type, config.bubtherm, config.Nt, config.medtherm,
        config.Mt, config.masstrans, config.rtol, config.atol,
    )
    p = problem.parameters
    tn = time_s / p["t0"]
    args = (
        p,
        config.stress,
        config.radial,
        config.bubtherm,
        problem.bubble_D1,
        problem.bubble_D2,
        problem.bubble_grid,
        config.medtherm,
        problem.medium,
        config.masstrans,
        _WallState(),
        problem.forcing,
    )
    started = perf_counter()
    solution = solve_ivp(
        _rhs,
        (tn[0], tn[-1]),
        problem.initial_state,
        t_eval=tn,
        args=args,
        events=_radius_floor_event,
        method="LSODA",
        rtol=config.rtol,
        atol=config.atol,
    )
    elapsed = perf_counter() - started
    complete = solution.y.shape[1] == time_s.size
    finite = bool(np.all(np.isfinite(solution.y)))
    success = bool(solution.success and complete and finite)
    message = str(solution.message)
    if solution.success and not complete:
        message = f"{message}; terminated before the final requested time"
    elif solution.success and not finite:
        message = f"{message}; solution contains non-finite states"
    stats = SolverStats(
        backend="scipy-lsoda",
        success=success,
        message=message,
        nfev=int(solution.nfev),
        njev=int(solution.njev),
        nlu=int(solution.nlu),
        elapsed_s=elapsed,
    )
    if not success:
        raise SimulationError(f"IMR integration failed: {message}", stats)
    return time_s, solution, stats


def _solve_prepared(problem: PreparedProblem, tv) -> SimulationResult:
    time_s, solution, stats = _integrate_prepared(problem, tv)
    return _build_result(problem, time_s, solution, stats)


def simulate(tv, config: SimulationConfig) -> SimulationResult:
    """Run one simulation and return immutable physical histories."""
    return prepare(config).solve(tv)
