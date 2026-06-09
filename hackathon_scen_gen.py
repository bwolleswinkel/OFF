"""Script to generate scenarios for the hackathon"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import yaml

if TYPE_CHECKING:
    from typing import Literal


# FROM: GitHub Copilot Gemini 3 Flash | 2026/06/09
class IndentedDumper(yaml.Dumper):
    """Custom YAML dumper that indents lists and dictionaries such that list items following keys 
    are indented (with 2 spaces) under the key, rather than at the same level"""
    def increase_indent(self, flow=False, indentless=False):
        # This force indentless=False to ensure lists are always indented
        return super(IndentedDumper, self).increase_indent(flow, False)


# FROM: GitHub Copilot Claude Haiku 4.5 | 2026/05/28
def generate_sequence(ranges, change_times, change_rates):
    """Generate value sequence with linear interpolation at change points."""
    current = np.random.uniform(*ranges[np.random.randint(len(ranges))])
    t_out, v_out = [0.0], [current]

    for i, t_change in enumerate(change_times):
        t_next = change_times[i + 1] if i + 1 < len(change_times) else timespan
        if t_change > t_out[-1]:
            t_out.append(float(t_change))
            v_out.append(current)

        target = np.random.uniform(*ranges[np.random.randint(len(ranges))])
        delta = target - current
        if np.isclose(delta, 0.0):
            t_out.append(float(t_next))
            v_out.append(current)
            continue

        t_reach = float(t_change + abs(delta) / change_rates[i])
        if t_reach < t_next:
            # Ramp to target, then plateau until next event.
            t_out.extend([t_reach, float(t_next)])
            v_out.extend([target, target])
            current = target
            continue

        # Not enough time to reach target before next event.
        current += np.sign(delta) * change_rates[i] * (t_next - t_change)
        t_out.append(float(t_next))
        v_out.append(current)

    if t_out[-1] < timespan:
        t_out.append(float(timespan))
        v_out.append(current)
    return t_out, v_out


# FROM: GitHub Copilot Claude Haiku 4.5 | 2026/05/28
def varying_profile(base_value, value_range, span_range, t_end, rng):
    """Create a piecewise-linear profile with random target updates over time."""
    t_points, v_points = [0.0], [float(base_value)]
    t_now = 0.0
    while t_now < t_end:
        dt = float(rng.uniform(*span_range))
        t_now = min(t_end, t_now + dt)
        t_points.append(t_now)
        v_points.append(float(rng.uniform(*value_range)))
    return np.array(t_points), np.array(v_points)


# FROM: GitHub Copilot Claude Haiku 4.5 | 2026/05/28
def add_varying_sine(trend_t, trend_v, dt, t_end, period_range, amp_base,
                     amp_range, span_range):
    """Add a sine perturbation with linearly varying period and amplitude."""
    rng = np.random.default_rng()
    t = np.arange(0.0, t_end + dt, dt)
    trend = np.interp(t, trend_t, trend_v)

    base_period = float(rng.uniform(*period_range))
    p_t, p_v = varying_profile(base_period, period_range, span_range, t_end, rng)
    a_t, a_v = varying_profile(amp_base, amp_range, span_range, t_end, rng)
    period = np.interp(t, p_t, p_v)
    amp = np.interp(t, a_t, a_v)

    phase = np.cumsum(2.0 * np.pi * dt / period)
    wave = amp * np.sin(phase)
    signal = trend + wave
    return t, signal.tolist()


if __name__ == '__main__':
    # Set the seed
    seed: int | None = 45538370

    # Set the output path
    path_to_out: Path = Path('run_amb.yaml')

    # Set parameters
    timestep: int = 4
    timespan: int = 10 * 60
    N_WT: int = 9
    RATED_POWER_MW: float = 5.0

    # Set the characteristics of the scenario
    wind_direction_overlap: Literal['overlap', 'no_overlap'] = 'no_overlap'
    wind_direction_change: Literal['change', 'no_change'] = 'change'
    wind_speed_rated: Literal['below_rated', 'at_rated', 'above_rated', 'all'] = 'all'
    wind_speed_change: Literal['change', 'no_change'] = 'change'
    turbulence_intensity: Literal['low', 'high'] = 'high'
    grid_demand: Literal['constant'] = 'constant'

    # === WIND FARM AND TURBINE ===

    layout = np.array([[ 600, 2400], [1500, 2400], [2400, 2400],
                       [ 600, 1500], [1500, 1500], [2400, 1500],
                       [ 600,  600], [1500,  600], [2400,  600]])
    
    cut_in_wind_speed = 3.0
    rated_wind_speed = 12.0
    cut_out_wind_speed = 25.0

    # === CONSTANTS ===

    wd_overlap = {'overlap': [[225, 240],
                              [265, 275],
                              [300, 315]],
                  'no_overlap': [[240, 265],
                                 [275, 300]]}
    ws_values = {'below_rated': [cut_in_wind_speed, rated_wind_speed - 1],
                 'at_rated': [rated_wind_speed - 1, rated_wind_speed + 1],
                 'above_rated': [rated_wind_speed + 1, cut_out_wind_speed - 5],
                 'all': [cut_in_wind_speed, cut_out_wind_speed - 5]}
    ti_values = {'low': [0.06, 0.1],
                 'high': [0.1, 0.2]}
    grid_values = {'constant': [5, N_WT * RATED_POWER_MW]}  # NOTE: In MW
    
    MIN_CHANGE_INTERVAL = timespan // 4

    WAVE_TIMESPAN = [60, 100]
    WAVE_PERIOD = [30, 200]
    WAVE_WD_AMPLITUDE = [1, 10]
    WAVE_WS_AMPLITUDE = [0.1, 1]
    
    # === GENERATE SCENARIO ===

    # Set the seed
    seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
    np.random.seed(seed)

    # Select the number of changes
    num_wd_changes = np.random.choice([0, 1, 2], p=([1, 0, 0]
                                                     if wind_direction_change == 'no_change'
                                                     else [0, 0.7, 0.3]))
    num_ws_changes = np.random.choice([0, 1, 2], p=([1, 0, 0]
                                                     if wind_speed_change == 'no_change'
                                                     else [0, 0.7, 0.3]))
    
    # Select the moments and rates of change
    wd_change_times = sorted(np.random.choice(range(MIN_CHANGE_INTERVAL, timespan - MIN_CHANGE_INTERVAL), num_wd_changes, replace=False))
    ws_change_times = sorted(np.random.choice(range(MIN_CHANGE_INTERVAL, timespan - MIN_CHANGE_INTERVAL), num_ws_changes, replace=False))
    wd_change_rates = [np.random.uniform(0.1, 0.5) for _ in range(num_wd_changes)]
    ws_change_rates = [np.random.uniform(0.01, 0.2) for _ in range(num_ws_changes)]
    
    # Generate wind direction and speed
    wd_t, wd = generate_sequence(wd_overlap[wind_direction_overlap], wd_change_times, wd_change_rates)
    ws_t, ws = generate_sequence([ws_values[wind_speed_rated]], ws_change_times, ws_change_rates)

    # Generate the turbulence intensity
    ti_val = np.random.uniform(*ti_values[turbulence_intensity])
    ti_t, ti = [0, timespan], [ti_val, ti_val]

    # Generate the grid demand
    grid_val = np.random.uniform(*grid_values[grid_demand])
    grid_t, grid = [0, timespan], [grid_val, grid_val]

    # === ADD TIME-VARYING VARIATIONS ===
    wd_t, wd = add_varying_sine(
        wd_t,
        wd,
        timestep,
        timespan,
        period_range=WAVE_PERIOD,
        amp_base=np.mean(WAVE_WD_AMPLITUDE),
        amp_range=WAVE_WD_AMPLITUDE,
        span_range=WAVE_TIMESPAN,
    )
    ws_t, ws = add_varying_sine(
        ws_t,
        ws,
        timestep,
        timespan,
        period_range=WAVE_PERIOD,
        amp_base=np.mean(WAVE_WS_AMPLITUDE),
        amp_range=WAVE_WS_AMPLITUDE,
        span_range=WAVE_TIMESPAN,
    )

    # === PLOT SCENARIO ===

    # Plot the wind direction and speed
    fig, (ax_wd, ax_ws, ax_ti, ax_grid) = plt.subplots(4, 1, sharex=True)
    ax_wd.plot(wd_t, wd)
    ax_wd.axhline(wd_overlap['no_overlap'][0][0], color='orange', linestyle='--')
    ax_wd.axhline(wd_overlap['no_overlap'][0][1], color='orange', linestyle='--')
    ax_wd.axhline(wd_overlap['no_overlap'][1][0], color='teal', linestyle='--')
    ax_wd.axhline(wd_overlap['no_overlap'][1][1], color='teal', linestyle='--')
    ax_wd.set_ylabel("Wind Direction (in °)")
    ax_wd.set_ylim(225, 315)
    ax_ws.plot(ws_t, ws)
    ax_ws.axhline(ws_values['below_rated'][0], color='green', linestyle='--')
    ax_ws.axhline(ws_values['below_rated'][1], color='green', linestyle='--')
    ax_ws.axhline(ws_values['above_rated'][0], color='purple', linestyle='--')
    ax_ws.axhline(ws_values['above_rated'][1], color='purple', linestyle='--')
    ax_ws.set_ylabel("Wind Speed (in m/s)")
    ax_ws.set_ylim(0, 25)
    ax_ti.plot(ti_t, ti)
    ax_ti.axhline(y=ti_values['low'][1], color='grey', linestyle='--')
    ax_ti.set_ylabel("Turbulence Intensity (in -)")
    ax_ti.set_ylim(ti_values['low'][0], ti_values['high'][1])
    ax_grid.plot(grid_t, np.array(grid) / N_WT)
    ax_grid.set_ylabel("Average Grid Demand (in in MW)")
    ax_grid.set_ylim(grid_values['constant'][0] / N_WT, grid_values['constant'][1] / N_WT)
    ax_grid.set_xlabel("Time (in seconds)")
    fig.suptitle(f"Scenario: "
                 f"{'Wake overlap' if wind_direction_overlap == 'overlap' else 'No wake overlap'}, "
                 f"{'change in wind direction' if wind_direction_change == 'change' else 'no change in wind direction'}, "
                 f"{'change in wind speed' if wind_speed_change == 'change' else 'no change in wind speed'}, "
                 f"{'below rated wind speed' if wind_speed_rated == 'below_rated' else 'at rated wind speed' if wind_speed_rated == 'at_rated' else 'above rated wind speed'}, "
                 f"{'low turbulence intensity' if turbulence_intensity == 'low' else 'high turbulence intensity'}, "
                 f"{'constant grid demand' if grid_demand == 'constant' else 'ERROR'}")
    plt.tight_layout()
    plt.show()

    # === SAVE TO YAML ===

    if True:
        #: Check if you want to save the data   
        parse = input(f"Do you want to save the data under the name '{path_to_out}'? (disable by setting this conditional to False): [y]es/[n]o ")
        if not parse.casefold() in ['y', 'yes']:
            pass
        else:
            data = {
                    'ambient': {
                        'name': "Automatically generated 10-min scenario",
                        'description': f"Synthetically generated scenario based on specified qualitative characteristics (wind_direction_overlap={wind_direction_overlap}, wind_direction_change={wind_direction_change}, wind_speed_rated={wind_speed_rated}, wind_speed_change={wind_speed_change}, turbulence_intensity={turbulence_intensity}, grid_demand={grid_demand} | seed={seed})",
                        'flow_field': {
                            'air_density': 1.225,
                            'turbulence_intensities': [round(ti[0], 2)],
                            'wind_directions': [round(elem, 1) for elem in wd],
                            'wind_directions_t': [int(elem) for elem in wd_t.tolist()],
                            'wind_speeds': [round(elem, 1) for elem in ws],
                            'wind_speeds_t': [int(elem) for elem in ws_t.tolist()],
                            'grid_demands': [int(grid[0] * 1E6)],
                            'wind_shear': 0.12,
                            'wind_veer': 0.0,
                            'corr_overwrite_direction': True,
                        }
                    }
                }
            
            yaml_string = yaml.dump(data, Dumper=IndentedDumper, default_flow_style=False, sort_keys=False)

            yaml_string = yaml_string.replace(
                'corr_overwrite_direction: true', 
                'corr_overwrite_direction: true  # All states are overwritten instead of only the first particle state'
            )

            yaml_string = yaml_string.replace('  flow_field:', '\n  flow_field:')

            with open('run_amb.yaml', 'w') as f:
                f.write(yaml_string)
