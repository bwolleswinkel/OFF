"""Script to test how the variables of downregulation surfaces behave"""

from pathlib import Path

import numpy as np
import scipy as sp
from floris import FlorisModel
import matplotlib.pyplot as plt

# Set the file name
file_path = Path('02_Examples_and_Cases') / '00_Inputs' / '00_OFF' / '05_Turbine' / 'NREL5MW' / 'Cp_Ct_NREL5MW_nrel.csv'

# Load the data
Data = np.loadtxt(file_path, delimiter=';', skiprows=1)
(pitch, tip_speed_ratio, Cp, Ct), stall = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)
pitch_range, tsr_range = [-10, 50], [0.05, 15]

# Create an interpolation function for Cp
Cp_func = lambda lbd_pitch, lbd_tsr: sp.interpolate.griddata(np.array((pitch.flatten(), tip_speed_ratio.flatten())).T, Cp.flatten(), (lbd_pitch, lbd_tsr))

# Construct a FLORIS model
fmodel = FlorisModel('02_Examples_and_Cases/00_Inputs/01_FLORIS/gch.yaml')
fmodel.set(turbine_type=['nrel_5MW'])

# Extract parameters
air_density, rotor_radius = fmodel.core.farm.turbine_map[0].power_thrust_table['ref_air_density'], fmodel.core.farm.turbine_map[0].rotor_radius

# Extract the wind speeds and power curve
u = fmodel.core.farm.turbine_map[0].power_thrust_table['wind_speed']
P_u = fmodel.core.farm.turbine_map[0].power_thrust_table['power'] * 1E3  # Convert to W

# Create an interpolation function for power
P_func = lambda lbd_u: sp.interpolate.interp1d(u, P_u, fill_value="extrapolate")(lbd_u)

# ------ FUNCTIONS ------


def tsr(omega_rpm: float, u: float) -> float:
    omega_rad = omega_rpm * (2 * np.pi) / 60
    return (omega_rad * rotor_radius) / u


def aero_torque(blade_pitch: float, u: float, omega_rpm: float) -> float:
    # Calculate aerodynamic power first
    omega_rad = omega_rpm * (2 * np.pi) / 60
    aero_power = (1 / 2) * air_density * np.pi * (rotor_radius ** 2) * (u ** 3) * Cp_func(blade_pitch, tsr(omega_rpm, u))
    return aero_power / omega_rad if omega_rad > 0 else 0.0


def gen_torque(omega_rpm: float, power_setpoint: float) -> float:
    omega_rad = omega_rpm * (2 * np.pi) / 60
    return power_setpoint / omega_rad


def obj(omega_rpm: float, blade_pitch: float, u: float, power_setpoint: float) -> float:
    return aero_torque(blade_pitch, u, omega_rpm) - gen_torque(omega_rpm, power_setpoint)


# ------ SCRIPT ------

# Select a wind speed and power setpoint
u_select = 7.0

print(f"Wind speed: {u_select} m/s")
print(f"Power setpoint: {P_func(u_select) * 1E-3:.1f} kW")

# Solve for the root using scipy fsolve
# NOTE: Here, the assumption is blade pitch is zero!
rotor_speed = sp.optimize.fsolve(obj, x0=10.0, args=(0.0, u_select, P_func(u_select)))

# Print the result
print(f"\nAt wind speed {u_select} m/s, the rotor speed is {rotor_speed[0]:.2f} RPM")

# Loop over a range of wind speeds to create a downregulation surface
rotor_speeds = []
for u_test in u:
    if u_test < 3.0 or u_test > 11.4:
        rotor_speeds.append(0.0)
        continue
    # TEMP
    #
    print(f"Testing wind speed: {u_test} m/s")
    print(f"Power setpoint: {P_func(u_test)/1000:.1f} kW\n")
    #
    rotor_speeds.append(sp.optimize.fsolve(obj, x0=10.0, args=(0.0, u_test, P_func(u_test)))[0])
    # TEMP
    #
    print(f"Rotor speed: {rotor_speeds[-1]:.2f} RPM\n")
    #

# ------ PLOTTING ------

fig_rotor_speed, ax_rotor_speed = plt.subplots()
ax_rotor_speed.plot(u, rotor_speeds, label=r"$\omega(u)$", color='blue')
ax_rotor_speed.set_xlabel("Wind Speed (m/s)")
ax_rotor_speed.set_ylabel("Rotor Speed (RPM)", color='blue')
ax_rotor_speed.tick_params(axis='y', labelcolor='blue')
ax_rotor_speed.legend(loc='upper left')

# Show the plot
plt.show()