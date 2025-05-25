# //////////////////////////////////////////////////////////////////// #
#    ____  ______ ______ 
#   / __ \|  ____|  ____|
#  | |  | | |__  | |__   
#  | |  | |  __| |  __|  
#  | |__| | |    | |     
#   \____/|_|    |_|     
# //////////////////////////////////////////////////////////////////// #

# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

# //////////////////////////////////////////////////////////////////// #
# Welcome to the example OFF main file. This showcases how to run a simulation using the OFF framework.
# The settings are defined in the run_example.yaml file, have a look to see what is possible.
# If you experience issues, create a new issue on the GitHub page https://github.com/TUDelft-DataDrivenControl/OFF
# //////////////////////////////////////////////////////////////////// #

import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time

def main():
    start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()

    # ====== BART ======

    from utils import tableau_color_palette_10 as col_vals

    input_file_name = 'run_example_3T_late_turning'

    # ====== BART ======

    # Create the input file
    path_input = f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/{input_file_name}.yaml'
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path(path_input)
    
    # One case used for the publication "A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies" Becker, Lejeune et al. 2024
    #oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/nawea_grid_zp_ki0-02_th5_LuT.yaml')
    
    # Run the simulation
    oi.run_sim()

    print("---OFF Simulation took %s seconds ---" % (time.time() - start_time))

    # Store output
    oi.store_measurements()
    oi.store_applied_control()
    oi.store_run_file()

    # ====== BART ======

    from pathlib import Path
    import yaml

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract the path name
    path_name_run = Path(oi.off_sim.sim_dir).name
    path_name_input = Path(path_input)

    # Read the wind direction as two lists
    # FROM: https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    with open(path_name_input) as stream:
        try:
            input_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract the time step, simulation time, and number of simulation steps
    # FIXME: I have no idea how robust this is, because I don't know how it calculates time time steps
    time_step, t_0, t_end = input_file['sim']['sim']['time step'], input_file['sim']['sim']['time start'], input_file['sim']['sim']['time end'],
    N_sim = int((t_end - t_0)/ time_step)
    t_range = np.arange(t_0, t_end, time_step)

    # Extract the number of wind turbines and the layout
    n_wt = len(input_file['wind_farm']['farm']['layout_x'])
    layout = np.column_stack((input_file['wind_farm']['farm']['layout_x'], input_file['wind_farm']['farm']['layout_y']))

    # Extract parameters from the wind turbine
    air_density = input_file['ambient']['flow_field']['air_density']
    rotor_diameter = input_file['turbine']['iea_10MW']['rotor_diameter']
    generator_efficiency = input_file['turbine']['iea_10MW']['generator_efficiency']
    u_power_coeffs, C_P_coeffs = np.array(input_file['turbine']['iea_10MW']['performance']['Cp_curve']['Cp_u_wind_speeds']), np.array(input_file['turbine']['iea_10MW']['performance']['Cp_curve']['Cp_u_values'])
    u_thrust_coeffs, C_T_coeffs = np.array(input_file['turbine']['iea_10MW']['performance']['Ct_curve']['Ct_u_wind_speeds']), np.array(input_file['turbine']['iea_10MW']['performance']['Ct_curve']['Ct_u_values'])

    # Create the ambient input file
    wd_input_file = [input_file['ambient']['flow_field']['wind_directions_t'], input_file['ambient']['flow_field']['wind_directions']]
    ws_input_file = [input_file['ambient']['flow_field']['wind_speeds_t'], input_file['ambient']['flow_field']['wind_speeds']]
    # FIXME: Can turbulence intensities not be time-dependent?
    ti_input_file = [[0], input_file['ambient']['flow_field']['turbulence_intensities']]

    # Create the control setpoints input files
    yaw_input_file = [input_file['controller']['settings']['orientation_t'], input_file['controller']['settings']['orientation_deg']]

    # Add the interpolation
    wd_ts = np.interp(t_range, wd_input_file[0], wd_input_file[1])
    ws_ts = np.interp(t_range, ws_input_file[0], ws_input_file[1])
    ti_ts = np.interp(t_range, ti_input_file[0], ti_input_file[1])

    # Add the power measurements as time-series
    measurements = pd.read_csv(f'runs/{path_name_run}/measurements.csv')
    control_applied = pd.read_csv(f'runs/{path_name_run}/applied_control.csv')

    # Extract the power
    power = [measurements.loc[measurements['t_idx'] == idx, 'power_OFF'] for idx in range(n_wt)]

    # Extract the C_P and C_T curver
    C_P = []
    C_T = [measurements.loc[measurements['t_idx'] == idx, 'Ct_FLORIS'] for idx in range(n_wt)]

    # Extract the local wind speed and local TI
    ws_local = [measurements.loc[measurements['t_idx'] == idx, 'u_abs_eff_FLORIS'] for idx in range(n_wt)]
    ti_local = [measurements.loc[measurements['t_idx'] == idx, 'TI_FLORIS'] for idx in range(n_wt)]

    # Extract the yaw angles, and actual orientation
    yaw_angles = [control_applied.loc[control_applied['t_idx'] == idx, 'yaw'] for idx in range(n_wt)]
    turbine_orientation = [control_applied.loc[control_applied['t_idx'] == idx, 'orientation'] for idx in range(n_wt)]

    # ------ PLOTTING ------

    # Set the plotting params
    plot_power_seperate = False

    # FIXME: For some reason, an empty plot is generated above? This does not seem to be caused by debug, but rather by the code I added?
    plt.close('all')

    # Plot the layout
    fig_layout, ax_layout = plt.subplots()
    ax_layout.plot(layout[:, 0], layout[:, 1], 'x')
    for idx in range(n_wt):
        ax_layout.text(layout[idx, 0], layout[idx, 1], f'{idx:02d}', fontsize=12, ha='left', va='bottom')
    fig_layout.suptitle("Layout of the wind farm")

    # Plot the C_P and C_T curves over wind speeds
    fig_power_thrust_coeffs, (ax_power_coeffs, ax_thrust_coeffs, ax_power_curve) = plt.subplots(3, 1)
    ax_power_coeffs.plot(u_power_coeffs, C_P_coeffs, color=col_vals[0], label=r'$C_\mathrm{P}(u_{\mathrm{eff}})$')
    ax_power_coeffs.legend(loc='upper right')
    ax_thrust_coeffs.plot(u_thrust_coeffs, C_T_coeffs, color=col_vals[1], label=r'$C_\mathrm{T}(u_{\mathrm{eff}})$')
    ax_thrust_coeffs.legend(loc='upper right')
    ax_power_curve.set_xlabel(r"Wind speed (in m/s)")
    fig_power_thrust_coeffs.suptitle("Power and thrust coefficients over wind speeds")
    
    # Plot the ambient wind direction, speed, and TI over time
    fig_ambient, (ax_ambient_wd, ax_ambient_ws, ax_ambient_ti) = plt.subplots(3, 1)
    ax_ambient_wd.plot(t_range, wd_ts, color='blue', label=r'Interpolated')
    ax_ambient_wd.plot(wd_input_file[0], wd_input_file[1], 'o', color='blue', markersize=2, label=r'Specified')
    ax_ambient_wd.set_ylabel(r"Direction $\theta$ (in °)")
    ax_ambient_wd_compass = ax_ambient_wd.twinx()
    ax_ambient_wd_compass.set_yticks(np.arange(0, 360 + 1, 45))
    ax_ambient_wd_compass.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
    ax_ambient_wd_compass.set_ylim(ax_ambient_wd.get_ylim())
    ax_ambient_wd.legend(loc='upper left', ncols=2)
    ax_ambient_ws.plot(t_range, ws_ts, color='green', label=r'Interpolated')
    ax_ambient_ws.plot(ws_input_file[0], ws_input_file[1], 'o', color='green', markersize=2, label=r'Specified')
    ax_ambient_ws.set_ylabel(r"Wind speed $U_{\infty}$ (in m/s)")
    ax_ambient_ws.legend(loc='upper left', ncols=2)
    ax_ambient_ti.plot(t_range, ti_ts * 100, color='red', label=r'Interpolated')
    ax_ambient_ti.plot(ti_input_file[0], [ti * 100 for ti in ti_input_file[1]], 'o', color='red', markersize=2, label=r'Specified')
    ax_ambient_ti.set_ylabel("Turbulence intensity (in %)")
    ax_ambient_ti.legend(loc='upper left', ncols=2)
    ax_ambient_ti.set_xlabel(r"Time $t$ (in s)")
    fig_ambient.suptitle("Ambient conditions")

    # Plot the control settings
    fig_control, (ax_yaw, ax_orientation, ax_powersetpoint, ax_status) = plt.subplots(4, 1)
    # # FIXME: Comment this out, as this does not have the correct values
    # #
    # for idx in range(n_wt):
    #     ax_yaw.plot([0, 200, 800, 1200], [[270 - yaw for yaw in yaws] for elem, yaws in enumerate([[270, 260, 250, 250], [270, 270, 270, 270], [270, 270, 270, 270]]) if elem == idx][0], '--', color=col_vals[idx], drawstyle='steps-post', alpha=0.5)
    #     ax_yaw.plot([0, 200, 800], [[270 - yaw for yaw in yaws] for elem, yaws in enumerate([[270, 260, 250], [270, 270, 270], [270, 270, 270]]) if elem == idx][0], 'o', color=col_vals[idx], drawstyle='steps-post', label=fr"Setpoint $\gamma_{{\mathrm{{ref}},{idx}}}$")
    # #
    for idx in range(n_wt):
        ax_yaw.plot(t_range, yaw_angles[idx], color=col_vals[idx], label=fr"Actual $\gamma_{idx}$")
    ax_yaw.set_ylabel(r"Angle $\gamma_{i}$ (in °)")
    ax_yaw.legend(loc='upper left', ncols=n_wt)
    ax_yaw.set_ylim([-30, 30])
    for idx in range(n_wt):
        ax_orientation.plot(t_range, turbine_orientation[idx], color=col_vals[idx], label=fr"$\phi_{idx}$")
    ax_orientation.set_ylabel(r"Orientation (in °)")
    ax_orientation.legend(loc='upper left', ncols=n_wt)
    ax_status.set_xlabel(r"Time $t$ (in s)")
    fig_control.suptitle("Control of each turbine")

    # Plot the local wind direction, speed, and TI over time
    fig_local, (ax_local_wd, ax_local_ws, ax_local_ti) = plt.subplots(3, 1)
    for idx in range(n_wt):
        ax_local_ws.plot(t_range, ws_local[idx], color=col_vals[idx], label=fr"Local $U_{{\infty}}^{idx}$")
    ax_local_ws.set_ylabel(r"Wind speed (in m/s)")
    ax_local_ws.legend(loc='upper left', ncols=n_wt)
    for idx in range(n_wt):
        ax_local_ti.plot(t_range, ti_local[idx] * 100, color=col_vals[idx], label=fr"Local $\mathrm{{TI}}^{idx}$")
    ax_local_ti.set_ylabel(r"Turbulence intensity (in %)")
    ax_local_ti.legend(loc='upper left', ncols=n_wt)
    fig_local.suptitle("Local inflow conditions")

    # Plot the effective C_P and C_T values
    fig_power_thrust_eff, (ax_power_coeffs_eff, ax_thrust_coeffs_eff) = plt.subplots(2, 1)
    for idx in range(n_wt):
        ax_thrust_coeffs_eff.plot(t_range, C_T[idx], label=f'WT{idx:02d}')
    ax_thrust_coeffs_eff.set_ylabel(r"$C_{\mathrm{T}}$ (in -)")
    ax_thrust_coeffs_eff.legend(loc='upper left', ncols=n_wt)
    ax_thrust_coeffs_eff.set_xlabel(r"Time $t$ (in s)")
    fig_power_thrust_eff.suptitle(r"Effective $C_{\mathrm{P}}$ and $C_{\mathrm{T}}$ values")

    # Plot the power
    if plot_power_seperate:
        fig_power, ax_power = plt.subplots(n_wt, 1, sharex=True)
        for idx in range(n_wt):
            ax_power[idx].plot(t_range, power[idx], color=col_vals[idx], label=f'Power {idx:02d}')
            ax_power[idx].legend(loc='upper left')
            ax_power[idx].set_ylim([0, 1.1 * max(power[idx])])
        ax_power[-1].set_xlabel(r"Time $t$ (in s)")
        fig_power.suptitle("Power of turbines")
    else:
        fig_power, ax_power = plt.subplots()
        for idx in range(n_wt):
            ax_power.plot(t_range, power[idx], label=f'Power {idx:02d}')
        ax_power.set_ylabel('Power (in W)')
        ax_power.legend()
        ax_power.set_xlabel(r"Time $t$ (in s)")
        fig_power.suptitle("Power of turbines")

    # Show the plots
    # plt.close(fig_layout)
    # plt.close(fig_power_thrust_coeffs)
    # plt.close(fig_ambient)
    # plt.close(fig_control)
    # plt.close(fig_local)
    # plt.close(fig_power_thrust_eff)
    # plt.close(fig_power)
    plt.show()

    # ====== BART ======


if __name__ == "__main__":
    main()

