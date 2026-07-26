import numpy as np
import pytest

from imr_fast import (
    Giesekus,
    InitialState,
    NeoHookeanKelvinVoigt,
    PhysicalParameters,
    SampledForcing,
    SimulationConfig,
    SimulationResult,
    Zener,
    prepare,
    simulate,
)


def base_config(**overrides):
    values = {
        "R0": 225e-6,
        "Req": 225e-6 / 6,
        "material": NeoHookeanKelvinVoigt(2500.0, 0.1),
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_simulate_matches_prepared_solver():
    times = np.linspace(0.0, 1.2e-4, 100)
    config = base_config()

    result = simulate(times, config)
    prepared = prepare(config).solve(times)

    assert isinstance(result, SimulationResult)
    np.testing.assert_array_equal(result.time_s, times)
    np.testing.assert_array_equal(result.radius_ratio, prepared.radius_ratio)
    np.testing.assert_allclose(result.radius_m, config.R0 * result.radius_ratio)


def test_structured_result_arrays_are_read_only():
    result = simulate(np.linspace(0.0, 1e-5, 10), base_config())

    for array in (
        result.time_s,
        result.radius_ratio,
        result.wall_velocity_m_s,
        result.internal_pressure_pa,
        result.stress_integral_pa,
    ):
        with pytest.raises(ValueError):
            array.flat[0] = 2.0


def test_prepared_problem_is_reusable_and_returns_active_fields():
    times = np.linspace(0.0, 2e-5, 25)
    config = base_config(
        material=Zener(2500.0, 0.1, relaxation_time_s=1e-6),
        bubtherm=1,
        medtherm=1,
        masstrans=1,
        vapor=1,
        Nt=9,
        Mt=7,
    )
    problem = prepare(config)

    first = problem.solve(times)
    second = problem.solve(times)

    np.testing.assert_array_equal(first.radius_ratio, second.radius_ratio)
    assert first.bubble_temperature_k.shape == (times.size, config.Nt)
    assert first.medium_temperature_k.shape == (times.size, config.Mt)
    assert first.vapor_mass_fraction.shape == (times.size, config.Nt)
    assert first.stress_state.shape == (times.size, 1)
    assert first.stats.success and first.stats.nfev > 0


def test_sampled_constant_forcing_matches_analytic_forcing():
    times = np.linspace(0.0, 2e-5, 30)
    analytic = simulate(times, base_config(pA=4e4))
    sampled = simulate(
        times,
        base_config(
            sampled_forcing=SampledForcing(
                time_s=(times[0], times[-1]),
                pressure_pa=(4e4, 4e4),
            )
        ),
    )

    np.testing.assert_array_equal(sampled.radius_ratio, analytic.radius_ratio)


def test_configurable_physics_and_initial_conditions_are_dimensional():
    times = np.linspace(0.0, 1e-6, 5)
    config = base_config(
        bubtherm=1,
        physics=PhysicalParameters(
            far_field_pressure_pa=9e4,
            medium_density_kg_m3=1000.0,
            polytropic_exponent=1.3,
        ),
        initial=InitialState(
            wall_velocity_m_s=2.0,
            internal_pressure_pa=1.5e5,
            bubble_temperature_k=310.0,
        ),
    )

    result = simulate(times, config)

    assert result.wall_velocity_m_s[0] == pytest.approx(2.0)
    assert result.internal_pressure_pa[0] == pytest.approx(1.5e5)
    assert result.bubble_temperature_k[0, 0] == pytest.approx(310.0)


def test_distributed_constitutive_state_uses_prepared_grid():
    model = Giesekus(
        viscosity_pa_s=0.1,
        relaxation_time_s=2e-6,
        retardation_time_s=4e-7,
        mobility=0.1,
        points=24,
    )
    result = simulate(
        np.linspace(0.0, 2e-6, 5),
        base_config(material=model),
    )

    assert result.stress_state.shape == (5, 2 * model.points)
    assert result.stress_reference_radius_ratio.shape == (model.points,)
    with pytest.raises(ValueError):
        result.stress_reference_radius_ratio[0] = 2.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"R0": 0.0}, "R0 must be finite and positive"),
        ({"radial": 6}, "radial must be one of"),
        ({"medtherm": 1}, "medtherm=1 requires bubtherm=1"),
        ({"masstrans": 1}, "masstrans=1 requires bubtherm=1"),
    ],
)
def test_config_rejects_invalid_inputs(override, message):
    with pytest.raises(ValueError, match=message):
        base_config(**override)


@pytest.mark.parametrize(
    "times",
    [
        [0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, np.nan],
    ],
)
def test_simulation_rejects_invalid_time_grids(times):
    with pytest.raises(ValueError):
        simulate(times, base_config())


def test_structured_api_requires_config():
    with pytest.raises(TypeError, match="SimulationConfig"):
        simulate([0.0, 1.0], object())
