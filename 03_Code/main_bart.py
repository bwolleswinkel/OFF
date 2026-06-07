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
#
# TODO: The `ObservationPoint` should be it's own (data)class, and should store the appropriate attributes. This would be a much cleaner interpretation. For speed/convenience, they could also be simply stored as lists.
# NOTE: For loads, we should really look into "Small Wind Turbine Technology," from Probst et al. (2011)
# NOTE: For loads, and chord lengths, we need to check out the following, where parameters are provided:
# FROM: https://ieawindsystems.github.io/windIO/main/source/how_to_build_a_turbine_model.html  # nopep8

import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time

# ====== BART ======
from typing import Literal
from utils import convert
# ====== BART ======

def main():

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()

    # ====== BART ======

    from utils import tableau_color_palette_10 as col_vals

    # run_1T_ss_wt_dynamics_loads
    # run_1T_var_wt_dynamics_loads
    # run_nine_turbine_marcus_revised
    # run_example_bart_wt_dynamics
    # run_3T_wt_dynamics_loads
    # run_schkortleben
    # run_schkortleben_shutdown
    # run_1T_ss_wt_dynamics_downregulation
    # run_1T_downregulation_simulink
    # run_schkortleben_downregulation
    # run_schkortleben_downregulation_half
    # run_3T_op_mode
    # run_1T_ss_above_rated
    # run_3T_ext_ctrl
    # run_3T_downreg
    # run_schkortleben_scen_1a
    input_file_name = 'run_3T_wt_dynamics_loads'  # NOTE: Without .yaml

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
    start_time = time.time()
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
    import matplotlib as mpl

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
    # FIXME: Here we assume a heterogeneous wind farm, all turbine types are the same
    turbine_type = input_file['wind_farm']['farm']['turbine_type'][0]

    # Extract parameters from the wind turbine
    air_density = input_file['ambient']['flow_field']['air_density']
    rotor_diameter = input_file['turbine'][turbine_type]['rotor_diameter']
    generator_efficiency = input_file['turbine'][turbine_type]['generator_efficiency']
    u_power_coeffs, C_P_coeffs = np.array(input_file['turbine'][turbine_type]['performance']['Cp_curve']['Cp_u_wind_speeds']), np.array(input_file['turbine'][turbine_type]['performance']['Cp_curve']['Cp_u_values'])
    u_thrust_coeffs, C_T_coeffs = np.array(input_file['turbine'][turbine_type]['performance']['Ct_curve']['Ct_u_wind_speeds']), np.array(input_file['turbine'][turbine_type]['performance']['Ct_curve']['Ct_u_values'])
    rated_power = input_file['turbine'][turbine_type]['performance'].get('rated_power', np.nan)

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

    # Extract the power setpoints
    try:
        power_setpoints = [measurements.loc[measurements['t_idx'] == idx, 'power_setpoint'].astype(float).to_numpy() for idx in range(n_wt)]
    except KeyError:
        power_setpoints = [np.full_like(t_range, np.nan) for idx in range(n_wt)]

    # Extract the C_P and C_T curver
    C_P = []
    C_T = [measurements.loc[measurements['t_idx'] == idx, 'Ct_FLORIS'] for idx in range(n_wt)]

    # Extract the local wind speed and local TI
    ws_local = [measurements.loc[measurements['t_idx'] == idx, 'u_abs_eff_FLORIS'] for idx in range(n_wt)]
    ti_local = [measurements.loc[measurements['t_idx'] == idx, 'TI_FLORIS'] for idx in range(n_wt)]

    # Extract the yaw angles, and actual orientation
    yaw_angles = [control_applied.loc[control_applied['t_idx'] == idx, 'yaw'] for idx in range(n_wt)]
    turbine_orientation = [control_applied.loc[control_applied['t_idx'] == idx, 'orientation'] for idx in range(n_wt)]

    # Extract the rotor speed
    try:
        rotor_speed = [measurements.loc[measurements['t_idx'] == idx, 'omega'] for idx in range(n_wt)]
    except KeyError:
        rotor_speed = [np.full_like(t_range, np.nan) for idx in range(n_wt)]

    # Extract the rotor speed setpoint
    try:
        rotor_speed_setpoint = [measurements.loc[measurements['t_idx'] == idx, 'omega_setpoint'] for idx in range(n_wt)]
    except KeyError:
        rotor_speed_setpoint = [np.full_like(t_range, np.nan) for idx in range(n_wt)]

    # Extract the operational modes
    try:
        operational_modes = [measurements.loc[measurements['t_idx'] == idx, 'operational_mode'].values for idx in range(n_wt)]
    except KeyError:
        operational_modes = [np.full_like(t_range, np.nan) for idx in range(n_wt)]

    # Extract the generator torque and pitch angle
    try:
        generator_torque = [measurements.loc[measurements['t_idx'] == idx, 'T_g'] for idx in range(n_wt)]
        pitch_angle = [measurements.loc[measurements['t_idx'] == idx, 'pitch'] for idx in range(n_wt)]
    except KeyError:
        generator_torque = None
        pitch_angle = None

    # Extract the loads on each turbine
    try:
        flapwise_bending_moment = [[measurements.loc[measurements['t_idx'] == idx, f'flapwise_bending_moment_blade_{blade_idx + 1}'] for blade_idx in range(3)] for idx in range(n_wt)]
        edgewise_bending_moment = [[measurements.loc[measurements['t_idx'] == idx, f'edgewise_bending_moment_blade_{blade_idx + 1}'] for blade_idx in range(3)] for idx in range(n_wt)]
        normal_force = [[measurements.loc[measurements['t_idx'] == idx, f'normal_force_blade_{blade_idx + 1}'] for blade_idx in range(3)] for idx in range(n_wt)]
        # Convert to pandas dataframe per turbine
        flapwise_bending_moment = [pd.concat(flapwise_bending_moment[idx], axis=1) for idx in range(n_wt)]
        edgewise_bending_moment = [pd.concat(edgewise_bending_moment[idx], axis=1) for idx in range(n_wt)]
        normal_force = [pd.concat(normal_force[idx], axis=1) for idx in range(n_wt)]
    except KeyError:
        flapwise_bending_moment = None
        edgewise_bending_moment = None
        normal_force = None

    # ------ PLOTTING ------

    # Set the plotting params
    plot_power_seperate = False

    # FIXME: There is this wierd offset notation, which I want to disable; this below does NOT work
    mpl.rcParams['axes.formatter.useoffset'] = False
    # mpl.rcParams['axes.formatter.use_locale'] = False
    # mpl.rcParams['axes.formatter.use_mathtext'] = False

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
    #     ax_yaw.plot([0, 200, 800, 1200], [[270 - yaw for yaw in yaws] for elem, yaws in enumerate([[270, 260, 250, 250], [270, 270, 270, 270], [270, 270, 270, 270]]) if elem == idx][0], '--', color=col_vals[idx % len(col_vals)], drawstyle='steps-post', alpha=0.5)
    #     ax_yaw.plot([0, 200, 800], [[270 - yaw for yaw in yaws] for elem, yaws in enumerate([[270, 260, 250], [270, 270, 270], [270, 270, 270]]) if elem == idx][0], 'o', color=col_vals[idx % len(col_vals)], drawstyle='steps-post', label=fr"Setpoint $\gamma_{{\mathrm{{ref}},{idx}}}$")
    # #
    for idx in range(n_wt):
        ax_yaw.plot(t_range, yaw_angles[idx], color=col_vals[idx % len(col_vals)], label=fr"Actual $\gamma_{idx}$")
    ax_yaw.set_ylabel(r"Angle $\gamma_{i}$ (in °)")
    ax_yaw.legend(loc='upper left', ncols=n_wt)
    ax_yaw.set_ylim([-30, 30])
    for idx in range(n_wt):
        ax_orientation.plot(t_range, turbine_orientation[idx], color=col_vals[idx % len(col_vals)], label=fr"$\phi_{idx}$")
    ax_orientation.set_ylabel(r"Orientation (in °)")
    ax_orientation.legend(loc='upper left', ncols=n_wt)
    ax_status.set_xlabel(r"Time $t$ (in s)")
    fig_control.suptitle("Control of each turbine")
    # Plot the operational mode
    mode_mapping = {
        'power_production': 1,
        'shutting_down': 2,
        'emergency_stop': 3,
        'parked': 4,
        'starting_up': 5,
    }
    for idx in range(n_wt):
        mode_values = []
        for mode in operational_modes[idx]:
            mode_values.append(mode_mapping.get(mode, np.nan))
        ax_status.plot(t_range, mode_values, color=col_vals[idx % len(col_vals)], label=f'Turbine {idx:02d}', linewidth=2, drawstyle='steps-post')
    ax_status.set_ylabel('Operational Mode')
    ax_status.set_yticks([1, 2, 3, 4, 5])
    ax_status.set_yticklabels(['Power Production', 'Shutting Down', 'Emergency Stop', 'Parked', 'Starting Up'])
    ax_status.legend(loc='upper left', ncols=n_wt)
    ax_status.grid(True, alpha=0.3)

    # Plot the local wind direction, speed, and TI over time
    fig_local, (ax_local_wd, ax_local_ws, ax_local_ti) = plt.subplots(3, 1)
    for idx in range(n_wt):
        ax_local_ws.plot(t_range, ws_local[idx], color=col_vals[idx % len(col_vals)], label=fr"Local $U_{{\infty}}^{idx}$")
    ax_local_ws.set_ylabel(r"Wind speed (in m/s)")
    ax_local_ws.legend(loc='upper left', ncols=n_wt)
    for idx in range(n_wt):
        ax_local_ti.plot(t_range, ti_local[idx] * 100, color=col_vals[idx % len(col_vals)], label=fr"Local $\mathrm{{TI}}^{idx}$")
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

    # Set the options for load plotting
    plot_turbine: Literal['seperate_plots', 'subplots'] = 'seperate_plots'

    match plot_turbine:
        case 'subplots':
            if flapwise_bending_moment is not None:
                fig_loads, axs_loads = plt.subplots(3, n_wt, sharex=True)
                for wt_idx in range(n_wt):
                    for blade_idx in range(3):
                        # BUG: If `axs_loads` has only one column, indexing fails; hence the workaround below
                        if n_wt == 1:
                            axs_loads[0].plot(t_range, flapwise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                            axs_loads[1].plot(t_range, edgewise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                            axs_loads[2].plot(t_range, normal_force[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                        else:
                            axs_loads[0, wt_idx].plot(t_range, flapwise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                            axs_loads[1, wt_idx].plot(t_range, edgewise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                            axs_loads[2, wt_idx].plot(t_range, normal_force[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                    if n_wt == 1:
                        axs_loads[0].set_title(f'Turbine {wt_idx + 1:02d}')
                        axs_loads[0].set_ylabel('Flapwise Bending Moment (in Nm)')
                        axs_loads[1].set_ylabel('Edgewise Bending Moment (in Nm)')
                        axs_loads[2].set_ylabel('Normal Force (in N)')
                        axs_loads[2].set_xlabel(r"Time $t$ (in s)")
                        axs_loads[0].legend(loc='upper left', ncols=1)
                    else:
                        axs_loads[0, wt_idx].set_title(f'Turbine {wt_idx:02d}')
                        axs_loads[0, wt_idx].set_ylabel('Flapwise Bending Moment (in Nm)')
                        axs_loads[1, wt_idx].set_ylabel('Edgewise Bending Moment (in Nm)')
                        axs_loads[2, wt_idx].set_ylabel('Normal Force (in N)')
                        axs_loads[2, wt_idx].set_xlabel(r"Time $t$ (in s)")
                        axs_loads[0, wt_idx].legend(loc='upper left', ncols=1)
                fig_loads.suptitle("Loads on each turbine")
            else:
                fig_loads = None
        case 'seperate_plots':
            if flapwise_bending_moment is not None:
                for wt_idx in range(n_wt):
                    fig_loads, (ax_flapwise, ax_edgewise, ax_normal) = plt.subplots(3, 1, sharex=True)
                    for blade_idx in range(3):
                        ax_flapwise.plot(t_range, flapwise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                        ax_edgewise.plot(t_range, edgewise_bending_moment[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                        ax_normal.plot(t_range, normal_force[wt_idx].iloc[:, blade_idx], color=col_vals[blade_idx], label=f'Blade {blade_idx + 1}')
                    ax_flapwise.set_ylabel('Flapwise Bending Moment (in Nm)')
                    ax_edgewise.set_ylabel('Edgewise Bending Moment (in Nm)')
                    ax_normal.set_ylabel('Normal Force (in N)')
                    ax_normal.set_xlabel(r"Time $t$ (in s)")
                    ax_flapwise.legend(loc='upper left', ncols=1)
                    fig_loads.suptitle(f"Loads on Turbine {wt_idx:02d}")
            else:
                fig_loads = None
        case _:
            raise ValueError(f"Invalid option for plot_loads: {plot_turbine}")
        
    # Plot the generator torque and pitch angle
    if generator_torque is not None:
        match plot_turbine:
            case 'subplots':
                fig_control_torque_pitch, axs_control = plt.subplots(2, n_wt, sharex=True)
                for wt_idx in range(n_wt):
                    if n_wt == 1:
                        axs_control[0].plot(t_range, generator_torque[wt_idx], color=col_vals[0], label='Generator Torque')
                        axs_control[1].plot(t_range, pitch_angle[wt_idx], color=col_vals[1], label='Pitch Angle')
                        axs_control[0].set_title(f'Turbine {wt_idx:02d}')
                        axs_control[0].set_ylabel('Generator Torque (in Nm)')
                        axs_control[1].set_ylabel('Pitch Angle (in °)')
                        axs_control[1].set_xlabel(r"Time $t$ (in s)")
                        axs_control[0].legend(loc='upper left')
                    else:
                        axs_control[0, wt_idx].plot(t_range, generator_torque[wt_idx], color=col_vals[0], label='Generator Torque')
                        axs_control[1, wt_idx].plot(t_range, pitch_angle[wt_idx], color=col_vals[1], label='Pitch Angle')
                        axs_control[0, wt_idx].set_title(f'Turbine {wt_idx:02d}')
                        axs_control[0, wt_idx].set_ylabel('Generator Torque (in Nm)')
                        axs_control[1, wt_idx].set_ylabel('Pitch Angle (in °)')
                        axs_control[1, wt_idx].set_xlabel(r"Time $t$ (in s)")
                        axs_control[0, wt_idx].legend(loc='upper left')
                fig_control_torque_pitch.suptitle("Generator torque and pitch angle for each turbine")
            case 'seperate_plots':
                for wt_idx in range(n_wt):
                    fig_control_torque_pitch, (ax_torque, ax_pitch) = plt.subplots(2, 1, sharex=True)
                    ax_torque.plot(t_range, generator_torque[wt_idx], color=col_vals[0], label='Generator Torque')
                    ax_pitch.plot(t_range, pitch_angle[wt_idx], color=col_vals[1], label='Pitch Angle')
                    ax_torque.set_ylabel('Generator Torque (in Nm)')
                    ax_pitch.set_ylabel('Pitch Angle (in °)')
                    ax_pitch.set_xlabel(r"Time $t$ (in s)")
                    ax_torque.legend(loc='upper left')
                    fig_control_torque_pitch.suptitle(f"Generator torque and pitch angle for Turbine {wt_idx:02d}")
            case _:
                raise ValueError(f"Invalid option for plot_turbine: {plot_turbine}")
    else:
        fig_control_torque_pitch = None

    if not np.isnan(rated_power):
        rated_tol = max(abs(rated_power) * 1e-3, abs(rated_power) * 1e-6)
        power_mask_plot = [np.isclose(ps, rated_power, rtol=1e-3, atol=rated_tol) for ps in power_setpoints]
        power_setpoints_plot = [np.where(mask, np.nan, ps) for mask, ps in zip(power_mask_plot, power_setpoints)]
        rotor_speed_setpoints_plot = [
            np.where(mask, np.nan, np.asarray(rs, dtype=float))
            for mask, rs in zip(power_mask_plot, rotor_speed_setpoint)
        ]
    else:
        power_setpoints_plot = power_setpoints
        rotor_speed_setpoints_plot = [np.asarray(rs, dtype=float) for rs in rotor_speed_setpoint]

    # Plot the rotor speeds
    fig_rotor_speed, ax_rotor_speed = plt.subplots()
    for idx in range(n_wt):
        ax_rotor_speed.plot(t_range, convert(rotor_speed[idx], 'rad/s', 'RPM'), color=col_vals[idx % len(col_vals)], label=f'WT{idx:02d}')
        rotor_label = f'WT{idx:02d} setpoint' if not np.all(np.isnan(rotor_speed_setpoints_plot[idx])) else None
        ax_rotor_speed.plot(t_range, convert(rotor_speed_setpoints_plot[idx], 'rad/s', 'RPM'), '--', color=col_vals[idx % len(col_vals)], label=rotor_label)
    ax_rotor_speed.set_ylabel(r"Rotor speed $\omega$ (in RPM)")
    ax_rotor_speed.legend(loc='upper left', ncols=n_wt)
    ax_rotor_speed.set_xlabel(r"Time $t$ (in s)")
    fig_rotor_speed.suptitle("Rotor speeds")

    # Plot the power
    if plot_power_seperate:
        fig_power, ax_power = plt.subplots(n_wt, 1, sharex=True)
        for idx in range(n_wt):
            plot_color = col_vals[idx % len(col_vals)]
            ax_power[idx].plot(t_range, power[idx] * 1E-6, color=plot_color, label=f'Power {idx:02d}')
            label = f'Power setpoint {idx:02d}' if not np.all(np.isnan(power_setpoints_plot[idx])) else None
            ax_power[idx].plot(t_range, power_setpoints_plot[idx] * 1E-6, '--', color=plot_color, label=label)
            ax_power[idx].legend(loc='upper left')
            ax_power[idx].set_ylim([0, 1.1 * max(power[idx] * 1E-6)])
        ax_power[-1].set_xlabel(r"Time $t$ (in s)")
        fig_power.suptitle("Power of turbines")
    else:
        fig_power, ax_power = plt.subplots()
        for idx in range(n_wt):
            plot_color = col_vals[idx % len(col_vals)]
            ax_power.plot(t_range, power[idx] * 1E-6, color=plot_color, label=f'Power {idx:02d}')
            label = f'Power setpoint {idx:02d}' if not np.all(np.isnan(power_setpoints_plot[idx])) else None
            ax_power.plot(t_range, power_setpoints_plot[idx] * 1E-6, '--', color=plot_color, label=label)
        ax_power.set_ylabel('Power (in MW)')
        ax_power.legend()
        ax_power.set_xlabel(r"Time $t$ (in s)")
        fig_power.suptitle("Power of turbines")

    # Plot the total 
    fig_total_power, ax_total_power = plt.subplots()
    total_power = np.sum(power, axis=0)
    ax_total_power.plot(t_range, total_power * 1E-6, '--', color='black', label='Total power')
    ax_total_power.set_ylabel('Power (in MW)')
    ax_total_power.legend()
    ax_total_power.set_xlabel(r"Time $t$ (in s)")
    fig_total_power.suptitle("Total power of the wind farm")

    # Show the plots
    # plt.close(fig_layout)
    # plt.close(fig_power_thrust_coeffs)
    # plt.close(fig_ambient)
    # plt.close(fig_control)
    # plt.close(fig_local)
    # plt.close(fig_power_thrust_eff)
    # plt.close(fig_power)
    # plt.close(fig_total_power)
    plt.show()

    # ====== BART ======


if __name__ == "__main__":
    main()

