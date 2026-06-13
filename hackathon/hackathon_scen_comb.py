"""Combine different scenarios"""

from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import yaml

from hackathon_scen_gen import IndentedDumper


if __name__ == '__main__':
    # Set the directories
    directory_path: Path = Path('hackathon')
    data_path: Path = Path('to_be_combined')
    path_to_out: Path = Path('run_amb_comb.yaml')

    # Load the files in the path
    files = []
    for file in (directory_path / data_path).glob('*.npy'):
        files.append(np.load(file))

    # Check if there are any files
    if not files:
        print(f"No files found in the specified directory '{directory_path / data_path}'.")
        exit(1)

    # Extract the time-scales
    time_scales = [file[1, 0] - file[0, 0] for file in files]

    # Construct the combined scenario
    t_vals = []
    for (idx, file) in enumerate(files):
        t_vals.append(file[:, 0] + idx * (time_scales[max([idx - 1, 0])] * file[:, 0].size))
    t_vals = np.concatenate(t_vals)
    wd = np.concatenate([file[:, 1] for file in files])
    ws = np.concatenate([file[:, 2] for file in files])
    ti, ti_t = [file[0, 3] for file in files], [idx * (time_scales[max([idx - 1, 0])] * file[:, 0].size) for (idx, file) in enumerate(files)]
    grid, grid_t = [file[0, 4] for file in files], [idx * (time_scales[max([idx - 1, 0])] * file[:, 0].size) for (idx, file) in enumerate(files)]

    # Construct the filtered, combined, scenarios
    low_pass_filter = sp.signal.butter(3, 0.1)
    wd_fil = sp.signal.filtfilt(*low_pass_filter, wd)
    ws_fil = sp.signal.filtfilt(*low_pass_filter, ws)
    ti_fil = [np.mean(ti) for _ in files]

    # === PLOT SCENARIO ===

    # Plot the wind direction and speed
    fig, (ax_wd, ax_ws, ax_ti, ax_grid) = plt.subplots(4, 1, sharex=True)
    ax_wd.plot(t_vals, wd)
    ax_wd.set_ylabel("Wind Direction (in °)")
    ax_wd.set_ylim(225, 315)
    ax_ws.plot(t_vals, ws)
    ax_ws.set_ylabel("Wind Speed (in m/s)")
    ax_ws.set_ylim(0, 25)
    ax_ti.plot(ti_t + [t_vals[-1]], ti + [ti[-1]])
    ax_ti.set_ylabel("Turbulence Intensity (in -)")
    ax_grid.plot(grid_t + [t_vals[-1]], grid + [grid[-1]])
    ax_grid.set_ylabel("Average Grid Demand (in in MW)")
    ax_grid.set_xlabel("Time (in seconds)")
    plt.tight_layout()

    # Plot the wind direction and speed
    fig, (ax_wd_fil, ax_ws_fil, ax_ti_fil, ax_grid_fil) = plt.subplots(4, 1, sharex=True)
    ax_wd_fil.plot(t_vals, wd_fil)
    ax_wd_fil.set_ylabel("Wind Direction (in °)")
    ax_wd_fil.set_ylim(225, 315)
    ax_ws_fil.plot(t_vals, ws_fil)
    ax_ws_fil.set_ylabel("Wind Speed (in m/s)")
    ax_ws_fil.set_ylim(0, 25)
    ax_ti_fil.plot(ti_t + [t_vals[-1]], ti_fil + [ti_fil[-1]])
    ax_ti_fil.set_ylabel("Turbulence Intensity (in -)")
    ax_grid_fil.plot(grid_t + [t_vals[-1]], grid + [grid[-1]], drawstyle='steps-post')
    ax_grid_fil.set_ylabel("Average Grid Demand (in in MW)")
    ax_grid_fil.set_xlabel("Time (in seconds)")
    plt.tight_layout()
    
    # Show the plots
    plt.show()

    # === SAVE TO YAML ===

    if True:
        #: Check if you want to save the data   
        parse = input(f"Do you want to save the data under the name '{directory_path / path_to_out}'? (disable by setting this conditional to False): [y]es/[n]o ")
        if not parse.casefold() in ['y', 'yes']:
            pass
        else:
            #: Save the data as yaml
            data = {
                    'ambient': {
                        'name': "Combined scenarios",
                        'description': "Multiple scenarios stitched together, filtered with a low-pass Butterworth filter (cutoff frequency of 0.1) and averaged turbulence intensity.",
                        'flow_field': {
                            'air_density': 1.225,
                            'turbulence_intensities': [round(ti[0].item(), 2)],
                            'wind_directions': [round(elem, 1).item() for elem in wd],
                            'wind_directions_t': [int(elem) for elem in t_vals.tolist()],
                            'wind_speeds': [round(elem, 1).item() for elem in ws],
                            'wind_speeds_t': [int(elem) for elem in t_vals.tolist()],
                            'requested_power': [int(elem) for elem in grid],
                            'requested_power_t': [elem.item() for elem in grid_t],
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

            with open(directory_path / path_to_out, 'w') as f:
                f.write(yaml_string)