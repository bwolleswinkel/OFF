"""This is a script to test wind-turbine dynamics, on a single turbine.

# FIXME: For some reason, tsr_opt =/= (omega_rated * rotor_radius) / u_rated, which is not correct.
# FIXME: The TSR is calculated with omega in rad/s, correct?
"""

import yaml

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# ------ PARAMETERS ------

# Set which model to use
wt_model = 'nrel_5MW'

# Set atmospheric parameters
air_density = 1.225  # kg/m^3, air density

# Select the mode
power_mode = 'lookup_table' 

# Set the local wind speed 
u_sim = [[0], [11.4]]  # NOTE: The first list is time points, the second list is wind speeds (in m/s) at those time points

# Set the blade pitch
pitch_sim = [[0], [0]]  # NOTE: The first list is time points, the second list is pitch angles (in deg) at those time points

# Set the initial rotor speed
omega_0 = 12.1  # RPM

# Set the simulation parameters
dt = 0.001  # s
T_sim = 100.0  # s

# ------ SCRIPT ------


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another."""
    match (from_unit, to_unit):
        case ('RPM', 'rad/s'):
            return value * ((2 * np.pi) / 60)
        case ('rad/s', 'RPM'):
            return value * (60 / (2 * np.pi))
        case _:
            raise ValueError(f"Unsupported conversion from '{from_unit}' to '{to_unit}'")
        

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
        inertia = float(input_file['hub_inertia_low_speed_shaft'])
        gearbox_ratio = input_file['gearbox_ratio']
        generator_efficiency = input_file['generator_efficiency'] 
        u_cut_in = input_file['performance']['cutin_wind_speed']
        u_rated = input_file['performance']['rated_wind_speed']
        rated_power = input_file['performance']['rated_power']
        omega_rated = convert(input_file['performance']['rated_rot_speed'], 'RPM', 'rad/s')
        # FIXME: I don't see how including the gearbox ratio (see "On the Analysis and Synthesis of Wind Turbine Side–Side Tower Load Control via Demodulation", Pamososuryo et al. (2024)) makes sense here
        gearbox_ratio = 1.0
    case _:
        raise ValueError(f"Unsupported wind turbine model '{wt_model}'")
    
# Load the Cp curve depending on the power mode
match power_mode:
    case 'wind_speed':
        # Load the Cp curve from the FLORIS model
        from floris import FlorisModel
        fmodel = FlorisModel('02_Examples_and_Cases/00_Inputs/01_FLORIS/gch.yaml')
        fmodel.set(turbine_type=['nrel_5MW'])
        u, P_u = fmodel.core.farm.turbine_map[0].power_thrust_table['wind_speed'], fmodel.core.farm.turbine_map[0].power_thrust_table['power']
        Cp_u = 2 * P_u / (air_density * np.pi * (rotor_radius ** 2) * (u ** 3) * generator_efficiency)
        # NOTE: Here we convert np.nan to 0, which might not be correct
        Cp_interp = lambda lbd_u: np.nan_to_num(np.interp(lbd_u, u, Cp_u))
    case 'lookup_table':
        # Load the Cp curve from a lookup table
        Data = np.loadtxt('02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv', delimiter=';', skiprows=1)
        (pitch_lut, tsr_lut, Cp, Ct), stall = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)
        pitch_range, tsr_range = [-10, 50], [0.05, 50]
        # FIXME: This part is INSANELY slow
        # Cp_interp = lambda lbd_pitch, lbd_tsr: np.nan_to_num(sp.interpolate.griddata(np.array((pitch_lut.flatten(), tsr_lut.flatten())).T, Cp.flatten(), (lbd_pitch, lbd_tsr)))
        Cp_func = sp.interpolate.RegularGridInterpolator((np.linspace(*tsr_range, num=300), np.linspace(*pitch_range, num=120)), Cp, method='nearest', bounds_error=False, fill_value=np.nan)
        Cp_interp = lambda lbd_pitch, lbd_tsr: Cp_func((lbd_pitch, lbd_tsr))
    case _:
        raise ValueError(f"Unsupported power mode '{power_mode}'")
    
# Create an interpolation for the wind speeds
u_interp = lambda t: np.interp(t, u_sim[0], u_sim[1])

# Create an interpolation for the wind speeds
pitch_interp = lambda t: np.interp(t, pitch_sim[0], pitch_sim[1])

# Calculate the controller gain
# FIXME: Here we have that tsr_opt is NOT equal to (omega_rated * rotor_radius) / u_rated
# K = 1 / (2 * (tsr_opt ** 3)) * air_density * np.pi * (rotor_radius ** 5) * Cp_opt
K = 1 / (2 * (((omega_rated * rotor_radius) / u_rated) ** 3)) * air_density * np.pi * (rotor_radius ** 5) * Cp_opt

# ------ SIMULATION ------

# Initialize the variables
omega, pitch, tsr, power = [], [], [], []

# Set the initial conditions
omega.append(convert(omega_0, 'RPM', 'rad/s'))  # Initial rotor speed in RPM

# Simulate the wind turbine dynamics
t = 0
while t < T_sim:
    #: Extract the current rotor speed
    omega_t, u_t, pitch_t = omega[-1], u_interp(t), pitch_interp(t)
    #: Set the arguments for the Cp interpolation
    match power_mode:
        case 'wind_speed':
            args = (u_t,)
        case 'lookup_table':
            #: Calculate the tip speed ratio
            tsr_t = (omega_t * rotor_radius) / u_t if u_t > 0 else 0
            args = (tsr_t, pitch_t)
        case _:
            raise ValueError(f"Unsupported power mode '{power_mode}'")
    #: Calculate the aerodynamic torque
    Cp_t = Cp_interp(*args)
    T_a = 1 / (2 * omega_t) * air_density * np.pi * (rotor_radius ** 2) * Cp_t * (u_t ** 3)
    #: Calculate the generator torque
    T_g = K * (omega_t ** 2)
    #: Calculate the rotor acceleration
    omega_dot = (T_a - gearbox_ratio * T_g) / inertia
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
t_range = np.linspace(0, T_sim, 100)  # Time range for plotting
ax_u.plot(t_range, u_interp(t_range), label=r"$u(t)$")
ax_u.set_xlabel(r"Time $t$ (in s)")
ax_u.set_ylabel(r"Wind Speed $u$ (in m/s)")
ax_u.legend(loc='upper right')

# Plot the rotor speed
fig_omega, ax_omega = plt.subplots()
ax_omega.plot(np.arange(0, len(omega)) * dt, [convert(o, 'rad/s', 'RPM') for o in omega], label=r"$\omega(t)$")
ax_omega.set_xlabel(r"Time $t$ (in s)")
ax_omega.set_ylabel(r"Rotor Speed $\omega$ (in RPM)")
ax_omega.legend(loc='upper right')  

# Plot the path of (pitch, tsr)
fig_pitch_tsr, ax_pitch_tsr = plt.subplots()
ax_pitch_tsr.plot(pitch, tsr, label=r"$(\beta(t), \lambda(t))$")
ax_pitch_tsr.set_xlabel(r"Blade Pitch $\beta$ (in °)")
ax_pitch_tsr.set_ylabel(r"Tip Speed Ratio $\lambda$ (in -)")
ax_pitch_tsr.legend(loc='upper right')   

# Plot the power output
fig_power, ax_power = plt.subplots()
ax_power.plot(np.arange(0, len(power)) * dt, [o * 1E-6 for o in power], label=r"$P(t)$")
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

