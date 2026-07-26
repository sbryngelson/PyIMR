import numpy as np
import pytest

from imr_fast import SimulationConfig, SimulationResult, simulate, simulate_result


def base_config(**overrides):
    values = {
        "R0": 225e-6,
        "Req": 225e-6 / 6,
        "G": 2500.0,
        "mu": 0.1,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_structured_api_matches_legacy_solver():
    times = np.linspace(0.0, 1.2e-4, 100)
    config = base_config()

    result = simulate_result(times, config)
    legacy = simulate(times, config.R0, config.Req, config.G, config.mu)

    assert isinstance(result, SimulationResult)
    np.testing.assert_array_equal(result.time_s, times)
    np.testing.assert_allclose(result.radius_ratio, legacy, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.radius_m, config.R0 * legacy)


def test_structured_result_arrays_are_read_only():
    result = simulate_result(np.linspace(0.0, 1e-5, 10), base_config())

    with pytest.raises(ValueError):
        result.radius_ratio[0] = 2.0
    with pytest.raises(ValueError):
        result.time_s[0] = 2.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"R0": 0.0}, "R0 must be finite and positive"),
        ({"stress": 6}, "stress must be one of"),
        ({"stress": 3}, "stress=3 requires lam1 > 0"),
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
        simulate_result(times, base_config())


def test_structured_api_requires_config():
    with pytest.raises(TypeError, match="SimulationConfig"):
        simulate_result([0.0, 1.0], object())
