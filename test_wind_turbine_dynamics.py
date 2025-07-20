"""This is a script to test wind-turbine dynamics, on a single turbine.

# FIXME: For some reason, tsr_opt =/= (omega_rated * rotor_radius) / u_rated, which is not correct.
# FIXME: The TSR is calculated with omega in rad/s, correct? From the Cp lookup table, tsr_opt = 7.55 seems to be the correct one.
# FIXME: The dynamic model from "Self-learning-based secure control of wind power generation systems under cyber threat: Ensuring prescribed performance" is not working correctly.

# TODO: Include the WT models from "Zero-Dynamics Attack on Wind Turbines and Countermeasures Using Generalized Hold and Generalized Sampler"
# TODO: Include the WT models from "Self-learning-based secure control of wind power generation systems under cyber threat: Ensuring prescribed performance"
# TODO: Include the displacement model from "Wind farm inertia forecasting accounting for wake losses, control strategies, and operational constraints"

"""

import yaml
import warnings
import sys, os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append('03_Code')
from utils import convert

# ------ PARAMETERS ------

# Set which model to use
wt_model = 'nrel_5MW'  # 'nrel_5MW' | 'cart_1_5MW'

# Set which dynamic model to use
dynamics_model = 'pamososuryo_drive_train_2025'  # 'pamososuryo_drive_train_2025' | 'mazare_drive_train_2025'

# Set atmospheric parameters
air_density = 1.225  # kg/m^3, air density

# Select the power mode
power_mode = 'lookup_table'  # 'wind_speed' | 'lookup_table' | 'closed_form'

# Set the controller mode
controller_mode = 'K_omega_squared'  # 'K_omega_squared' | 'K_omega_cubed' | 'greedy_lio'

# Set the local wind speed 
# u_sim = [[0, 50, 50.1,], [11, 11, 4.0]]  # NOTE: The first list is time points, the second list is wind speeds (in m/s) at those time points
u_sim = [[0, 600], [8.0, 12.0]]

# Set the blade pitch
pitch_sim = [[0], [1]]  # NOTE: The first list is time points, the second list is pitch angles (in deg) at those time points
pitch_mode = 'lookup_table'  # 'zero' | 'lookup_table' | 'lookup_table_rotor_speed' | 'lookup_table_exponential' | 'reference'

# Set the initial rotor speed
omega_0 = 9.0  # RPM

# Set the simulation parameters
dt = 0.2  # s
T_sim = 1200.0  # s

# ------ SCRIPT ------

# Set the controller gains
K_P_gen = 1E12

# Load the data based on the model
match wt_model:
    case 'nrel_5MW':
        with open('02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/NREL_5MW.yaml', 'r') as stream:
            try:
                input_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        rotor_radius = input_file['rotor_diameter'] / 2
        Cp_opt = input_file['performance']['Cp_opt']
        tsr_opt = input_file['performance']['tsr_opt']
        # FIXME: I have no idea if this value is correct
        inertia = float(input_file['hub_inertia_low_speed_shaft'])
        # FIXME: I don't see how including the gearbox ratio (see "On the Analysis and Synthesis of Wind Turbine Side–Side Tower Load Control via Demodulation", Pamososuryo et al. (2024)) makes sense here
        gearbox_ratio = input_file['gearbox_ratio']
        generator_efficiency = input_file['generator_efficiency'] 
        u_cut_in = input_file['performance']['cutin_wind_speed']
        u_rated = input_file['performance']['rated_wind_speed']
        rated_power = input_file['performance']['rated_power']
        omega_rated = convert(input_file['performance']['rated_rot_speed'], 'RPM', 'rad/s')
        rated_gen_tor_torque = input_file['performance']['rated_gen_tor_torque'] 
        blade_pitch, blade_pitch_u = input_file['performance']['pitch']['pitch_curve']['pitch_u_values'], input_file['performance']['pitch']['pitch_curve']['pitch_u_wind_speeds']
        # FIXME: This value is from "On the Analysis and Synthesis of Wind Turbine Side–Side Tower Load Control via Demodulation", Pamososuryo et al. (2024), but I don't know if it is correct
        inertia = 4.0802E7  # kg*m^2
    case 'cart_1_5MW':
        rotor_radius = 35.0  # m
        J_r = 2.96E6  # kg*m^2, rotor inertia
        J_g = 53  # kg*m^2, generator inertia
        K_s = 5.6E9  # N*m/rad, drive-train spring factor
        D_s = 1E7  # N*m*s/rad, drive-train damping factor
        N_g = 87.965  # gearbox ratio
        generator_efficiency = 0.95  # generator efficiency
        omg_n = 11.11  # natural frequency of the pitch actuator (in rad/s)
        eta = 0.6  # damping ratio of the pitch actuator
        rated_power = 1.5E6  # W, rated power
        omega_rated = convert(2.1428, 'RPM', 'rad/s')  # rated rotor speed (in rad/s)
        # FIXME: This value is not provided in the paper
        u_rated = 8.4  # m/s, rated wind speed
        # FIXME: All values below are not defined
        Cp_opt = 0.4  # m/s, rated wind speed
        tsr_opt = np.nan
    case _:
        raise ValueError(f"Unsupported wind turbine model '{wt_model}'")
    
# Load the Cp curve depending on the power mode
match power_mode:
    case 'wind_speed':
        # Load the Cp curve from the FLORIS model
        from floris import FlorisModel
        fmodel = FlorisModel('02_Examples_and_Cases/00_Inputs/01_FLORIS/gch.yaml')
        fmodel.set(turbine_type=['nrel_5MW'])
        u, P_u = fmodel.core.farm.turbine_map[0].power_thrust_table['wind_speed'], fmodel.core.farm.turbine_map[0].power_thrust_table['power'] * 1E3  # Convert to W
        Cp_u = 2 * P_u / (air_density * np.pi * (rotor_radius ** 2) * (u ** 3) * generator_efficiency)
        # NOTE: Here we convert np.nan to 0, which might not be correct
        Cp_interp = lambda lbd_u: np.nan_to_num(np.interp(lbd_u, u, Cp_u))
    case 'lookup_table':
        # Load the Cp curve from a lookup table
        Data = np.loadtxt('02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv', delimiter=';', skiprows=1)
        (pitch_lut, tsr_lut, Cp, Ct), stall = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)
        pitch_range, tsr_range = [-10, 50], [0.05, 15]
        Cp_func = sp.interpolate.RegularGridInterpolator((np.linspace(*tsr_range, num=300), np.linspace(*pitch_range, num=120)), Cp, method='linear', bounds_error=False, fill_value=np.nan)
        Cp_interp = lambda lbd_pitch, lbd_tsr: Cp_func((lbd_pitch, lbd_tsr))
    case 'closed_form':
        tsr_star = lambda eval_tsr, eval_pitch: 1 / (eval_tsr + 0.08 * eval_pitch) - 0.035 / (eval_pitch ** 3 + 1)
        Cp_interp = lambda eval_tsr, eval_pitch: 0.22 * (116 * tsr_star(eval_tsr, eval_pitch) - 0.4 * eval_pitch - 5) * np.exp(-12.5 * tsr_star(eval_tsr, eval_pitch))
    case _:
        raise ValueError(f"Unsupported power mode '{power_mode}'")
    
# Create an interpolation for the wind speeds
u_interp = lambda t: np.interp(t, u_sim[0], u_sim[1])

# Create an interpolation for the wind speeds
pitch_ref_interp = lambda t: np.interp(t, pitch_sim[0], pitch_sim[1])

# Calculate the controller gain
# FIXME: Here we have that tsr_opt is NOT equal to (omega_rated * rotor_radius) / u_rated
# K = 1 / (2 * (tsr_opt ** 3)) * air_density * np.pi * (rotor_radius ** 5) * Cp_opt
K = 1 / (2 * (((omega_rated * rotor_radius) / u_rated) ** 3)) * air_density * np.pi * (rotor_radius ** 5) * Cp_opt

# ------ SIMULATION ------

# Initialize the variables
omega, omega_gen, theta, pitch, pitch_dot, tsr, power = [], [], [], [], [], [], []

# Set the initial conditions
omega.append(convert(omega_0, 'RPM', 'rad/s'))  # Initial rotor speed in RPM
omega_gen.append(100)  # Initial generator speed in rad/s
theta.append(0)  # Initial generator speed in rad/s
pitch_dot.append(0)  # Initial pitch angle in rad

# Simulate the wind turbine dynamics
t = 0
while t < T_sim:
    match dynamics_model:
        case 'pamososuryo_drive_train_2025':
            #: Extract the current rotor speed
            omega_t, u_t = omega[-1], u_interp(t)
            #: Extract the current pitch angle
            match pitch_mode:
                case 'zero':
                    pitch_t = 0
                case 'lookup_table':
                    pitch_t = np.interp(u_t, blade_pitch_u, blade_pitch)
                case 'lookup_table_rotor_speed':
                    # FIXME: This 'works', but gives very irregular 'chattering' behavior
                    if omega_t > omega_rated:
                        pitch_t = np.interp(u_t, blade_pitch_u, blade_pitch)
                    else:
                        pitch_t = 0
                case 'lookup_table_exponential':
                    #: Calculate the steady-state pitch
                    pitch_ss = np.interp(u_t, blade_pitch_u, blade_pitch)
                    #: Check the difference in rotor speed and steady-state rotor speed
                    delta_omega = omega_t - omega_rated
                    #: Calculate the proportional pitch angle
                    pitch_t = np.exp(1E1 * delta_omega) * pitch_ss
                case 'reference':
                    pitch_t = pitch_ref_interp(t)
            pitch_t = convert(pitch_t, 'deg', 'rad')  # Convert pitch
            #: Calculate the tip speed ratio
            tsr_t = (omega_t * rotor_radius) / u_t if u_t > 0 else 0
            #: Set the arguments for the Cp interpolation
            match power_mode:
                case 'wind_speed':
                    args = (u_t,)
                case 'lookup_table':
                    args = (tsr_t, convert(pitch_t, 'rad', 'deg'))
                case _:
                    raise ValueError(f"Unsupported power mode '{power_mode}'")
            #: Calculate the aerodynamic torque
            Cp_t = Cp_interp(*args)
            # FIXME: Sometimes this can be np.nan, which is problematic...
            if np.isnan(Cp_t):
                warnings.warn(f"Cp value is NaN at time {t:.2f} s, using 0 instead")
                Cp_t = 0
            T_a = 1 / (2 * omega_t) * air_density * np.pi * (rotor_radius ** 2) * Cp_t * (u_t ** 3)
            #: Calculate the generator torque
            match controller_mode:
                case 'K_omega_squared':
                    T_g = K * (omega_t ** 2)
                case 'K_omega_cubed':
                    T_g = K * (omega_t ** 3)
                case 'greedy_lio':
                    #: Calculate the rotor setpoint
                    if u_t < u_rated:
                        omega_setpoint = (tsr_opt * u_t) / rotor_radius
                    else:
                        omega_setpoint = omega_rated
                    #: Calculate the generator torque
                    if u_t < u_rated:
                        T_g = K_P_gen * (omega_t - omega_setpoint)
                    else:
                        T_g = rated_gen_tor_torque
                    # FIXME: Cap the generator torque to be positive
                    T_g = np.max([T_g, 0])
                case _:
                    raise ValueError(f"Unsupported controller mode '{controller_mode}'")
            #: Calculate the rotor acceleration
            omega_dot = (T_a - T_g) / inertia
            #: Calculate the power output
            # FIXME: Does the generator efficiency need to be included here?
            power_t = T_g * generator_efficiency * omega_t  # Power output in Watts
            #: Calculate the new rotor speed
            omega_new = omega_t + omega_dot * dt
            #: Save the new rotor speed
            omega.append(omega_new)
            pitch.append(pitch_t)
            tsr.append(tsr_t)
            power.append(power_t)
        case 'mazare_drive_train_2025':
            #: Extract the current variables speed
            omega_t, omega_gen_t, theta_t, u_t, pitch_t, pitch_dot_t, pitch_ref_t = omega[-1], omega_gen[-1], theta[-1], u_interp(t), pitch_ref_interp(t), pitch_dot[-1], pitch_ref_interp(t)
            #: Calculate the tip speed ratio
            tsr_t = (omega_t * rotor_radius) / u_t if u_t > 0 else 0
            #: Set the arguments for the Cp interpolation
            match power_mode:
                case 'closed_form':
                    args = (tsr_t, pitch_t)
                case _:
                    raise ValueError(f"Unsupported power mode '{power_mode}'")
            #: Calculate the aerodynamic power
            Cp_t = Cp_interp(*args)
            P_aero = 1 / 2 * np.pi * (rotor_radius ** 2) * (u_t ** 3) * Cp_t
            #: Calculate the generator torque
            match controller_mode:
                case 'K_omega_squared':
                    T_g = K * (omega_t ** 2)
                case 'K_omega_cubed':
                    T_g = K * (omega_t ** 3)
                case 'greedy_lio':
                    if u_t < u_rated:
                        T_g = K_P_gen * (omega_t - (tsr_opt * u_t) / rotor_radius)
                    else:
                        T_g = rated_gen_tor_torque
                case _:
                    raise ValueError(f"Unsupported controller mode '{controller_mode}'")
            #: Calculate the rotor acceleration, generator acceleration, decoupling angle acceleration, and blade pitch angle acceleration
            omega_dot = (P_aero / omega_t - omega_t * D_s + (omega_gen_t * D_s) / N_g - theta_t * K_s) / J_r
            omega_gen_dot = ((omega_t * D_s) / N_g - (omega_gen_t * D_s) / N_g ** 2 + (theta_t * K_s) / N_g - T_g) / J_g
            theta_dot = (omega_t - omega_gen_t / N_g)
            pitch_ddot = -2 * eta * omg_n * pitch_dot_t - omg_n ** 2 * pitch_t + omg_n ** 2 * pitch_ref_t
            #: Calculate the new rotor speed
            omega_new = omega_t + omega_dot * dt
            omega_gen_new = omega_gen_t + omega_gen_dot * dt
            theta_new = theta_t + theta_dot * dt
            pitch_new = pitch_t + pitch_dot_t * dt + 0.5 * pitch_ddot * dt ** 2
            pitch_dot_new = pitch_dot_t + pitch_ddot
            #: Calculate the power output
            power_t = T_g * generator_efficiency * omega_t  # Power output in Watts
            if np.isnan(power_t):
                warnings.warn(f"Power output is NaN at time {t:.2f} s, Cp value is {Cp_t:.3f}")
            #: Save the new state variables
            omega.append(omega_new)
            omega_gen.append(omega_gen_new)
            theta.append(theta_new)
            pitch.append(pitch_new)
            pitch_dot.append(pitch_dot_new)
            # Save the tip speed ratio and power output
            tsr.append(tsr_t)
            power.append(power_t)
        case _:
            raise ValueError(f"Unsupported dynamics model '{dynamics_model}'")
    #: Update the time
    t += dt

# ------ PRINTING ------

# Check the rated power
print(f"--- RATED POWER ---")
print(f"Rated power (from file): {rated_power * 1E-6:.2f} (in MW), P_aero: {1/2 * air_density * np.pi * (rotor_radius ** 2) * Cp_opt * (u_rated ** 3) * generator_efficiency * 1E-6:.2f} (in MW), P_gen (K * ω^2): {(K * omega_rated ** 2) * omega_rated * generator_efficiency * 1E-6:.2f} (in MW)")
print(f"Cp_opt (from file): {Cp_opt:.3f}, u_rated (from file): {u_rated:.1f}, omega_rated (from file): {convert(omega_rated, 'rad/s', 'RPM'):.1f} (in RPM), tsr_opt (from file): {tsr_opt:.2f}, tsr_calc = (omega_rated * R) / u_rated: {(omega_rated * rotor_radius) / u_rated:.2f}")
match power_mode:
    case 'wind_speed':
        print(f"Cp_opt: (from file): {Cp_opt:.3f}, Cp_opt (from wind speed): {Cp_interp(u_rated):.3f}")
    case 'lookup_table':
        print(f"Cp_opt: (from file): {Cp_opt:.3f}, Cp_opt (from lookup table, tsr_opt): {Cp_interp(tsr_opt, 0):.3f}, Cp_opt (from lookup table, tsr_calc): {Cp_interp((omega_rated * rotor_radius) / u_rated, 0):.3f}")
    case 'closed_form':
        pass
    case _:
        raise ValueError(f"Unsupported power mode '{power_mode}'")

# ------ PLOTTING ------

# Plot the Cp curve
if power_mode == 'wind_speed':
    fig_Cp_u, ax_Cp_u = plt.subplots()
    u_range = np.linspace(0, 25, 100)  # Wind speed range for plotting
    ax_Cp_u.plot(u_range, Cp_interp(u_range), label=r"$C_{P}(u)$")
    ax_Cp_u.axhline(Cp_opt, color='red', linestyle='--', label=r"$C_{P,opt}$")
    ax_Cp_u.set_xlabel(r"Wind Speed $u$ (in m/s)")
    ax_Cp_u.legend(loc='upper right')
else:
    fig_Cp_u, ax_Cp_u = None, None

# Plot the local wind speed
fig_u, ax_u = plt.subplots()
t_range = np.arange(0, T_sim + dt, dt)  # Time range for plotting
ax_u.plot(t_range, u_interp(t_range), label=r"$u(t)$")
ax_u.set_xlabel(r"Time $t$ (in s)")
ax_u.set_ylabel(r"Wind Speed $u$ (in m/s)")
ax_u.legend(loc='upper right')

# Plot the rotor speed
fig_omega, ax_omega = plt.subplots()
ax_omega.plot(t_range, [convert(o, 'rad/s', 'RPM') for o in omega], label=r"$\omega(t)$")
ax_omega.set_xlabel(r"Time $t$ (in s)")
ax_omega.set_ylabel(r"Rotor Speed $\omega$ (in RPM)")
ax_omega.legend(loc='upper right')  

# Plot the path of (pitch, tsr)
fig_pitch_tsr, ax_pitch_tsr = plt.subplots()
ax_pitch_tsr.imshow(np.flipud(Cp), cmap='inferno', extent=(pitch_range[0], pitch_range[1], tsr_range[0], tsr_range[1]), aspect='auto', origin='lower')
ax_pitch_tsr.plot([convert(elem, 'rad', 'deg') for elem in pitch], tsr, label=r"$(\beta(t), \lambda(t))$")
ax_pitch_tsr.set_xlabel(r"Blade Pitch $\beta$ (in °)")
ax_pitch_tsr.set_ylabel(r"Tip Speed Ratio $\lambda$ (in -)")
ax_pitch_tsr.legend(loc='upper right')   

# Plot the power output
fig_power, ax_power = plt.subplots()
ax_power.plot(t_range[:-1], [o * 1E-6 for o in power], label=r"$P(t)$")
ax_power.set_xlabel(r"Time $t$ (in s)")
ax_power.set_ylabel(r"Power Output $P$ (in MW)")
ax_power.legend(loc='upper right')

# Show the plots
# plt.close(fig_Cp_u)
# plt.close(fig_u)
# plt.close(fig_omega)
# plt.close(fig_pitch_tsr)
# plt.close(fig_power)
plt.show()

