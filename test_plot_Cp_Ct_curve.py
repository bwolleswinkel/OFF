"""This is a script to read and plot Cp and Ct curves from a file.

"""

import numpy as np
import scipy as sp
from floris import FlorisModel
import matplotlib.pyplot as plt

# ------ PARAMETERS ------

# Select the file to read
file_name = '02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv'  # '02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_mulder.csv' | '02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv'

# Select the downregulation setting
u_down = 9.0  # m/s
factor_down = 0.5  # Downregulation factor (factor_down * 100% of available power)

# ------ SCRIPT ------

# Load the data
match file_name:
    case '02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_mulder.csv':
        Data = np.loadtxt(file_name, delimiter=';', skiprows=1)
        pitch, tsr, Cp, Ct, stall = [np.flipud(Data[:, i].reshape(33, 51, order='F')) for i in [1, 2, 3, 4, -1]]
        pitch_range, tsr_range = [-5, 20], [2, 10]
    case '02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv':
        Data = np.loadtxt(file_name, delimiter=';', skiprows=1)
        (pitch, tsr, Cp, Ct), stall = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)
        pitch_range, tsr_range = [-10, 50], [0.05, 15]
    case _:
        raise ValueError(f"Unsupported file: {file_name}")
    
# Construct a FLORIS model
fmodel = FlorisModel('02_Examples_and_Cases/00_Inputs/01_FLORIS/gch.yaml')
fmodel.set(turbine_type=['nrel_5MW'])

# Extract the thrust coefficient
u = fmodel.core.farm.turbine_map[0].power_thrust_table['wind_speed']
Ct_u = fmodel.core.farm.turbine_map[0].power_thrust_table['thrust_coefficient']
P_u = fmodel.core.farm.turbine_map[0].power_thrust_table['power'] * 1E3  # Convert to W

# Set some parameters
air_density, rotor_radius = 1.225, fmodel.core.farm.turbine_map[0].rotor_radius
generator_efficiency = 0.994

# Calculate the power coefficient
Cp_u = 2 * P_u / (air_density * np.pi * (rotor_radius ** 2) * (u ** 3) * generator_efficiency)
  
# Create an interpolation function for Cp and Ct
Cp_interp = lambda lbd_pitch, lbd_tsr: sp.interpolate.griddata(np.array((pitch.flatten(), tsr.flatten())).T, Cp.flatten(), (lbd_pitch, lbd_tsr))
Ct_interp = lambda lbd_pitch, lbd_tsr: sp.interpolate.griddata(np.array((pitch.flatten(), tsr.flatten())).T, Ct.flatten(), (lbd_pitch, lbd_tsr))

# Find the maximum Cp value
# indices_max_value = np.nanargmax(Cp)
# pitch_max_value, tsr_max_value, Cp_max_value =  Cp[indices_max_value]
    
# Calculate the available and downregulation power
P_down_available = np.interp(u_down, u, P_u)
P_down = factor_down * P_down_available
Cp_down_available = 2 * P_down_available / (air_density * np.pi * (rotor_radius ** 2) * (u_down ** 3) * generator_efficiency)
Cp_down = 2 * P_down / (air_density * np.pi * (rotor_radius ** 2) * (u_down ** 3) * generator_efficiency)

# Find the values where this Cp is reached
pitch_interp, tsr_interp = np.meshgrid(np.linspace(*pitch_range, 500), np.linspace(*reversed(tsr_range), 500), indexing='xy')
Cp_agrees = (np.abs(Cp_interp(pitch_interp, tsr_interp) - Cp_down) <= 0.001).astype(float)
Cp_agrees[~Cp_agrees.astype(bool)] = np.nan
# FIXME: There are already two solutions to steady-state operation
tsr_lambda_available = np.linspace(*tsr_range, 500)[sp.signal.argrelextrema(np.abs(Cp_interp(0, np.linspace(*tsr_range, 500)) - Cp_down_available), np.less)[0]]

# Convert the tip speed ratios to rotor speed, for this downregulated wind speed
rotor_speed_down = (u_down * np.array(tsr_range)) / rotor_radius * (60 / (2 * np.pi))  # Convert to RPM

# ------ PRINTING ------

# Print the point of downregulation
print("--- Downregulation ---")
print(f"Wind speed: {u_down}, downregulation: {factor_down * 100}% (Available power: {P_down_available:.0f} -> {P_down:.0f})")

# ------ PLOTTING ------

# Plot the Cp and Ct curves indexed by wind speed
fig_cp_ct_u, (ax_power_u, ax_cp_u, ax_ct_u) = plt.subplots(3, 1, sharex=True)
ax_power_u.plot(u, P_u * 1E-6, label="Power $P(u)$")
ax_power_u.axvline(u_down, color='red', linestyle='--', label=f"Downregulation at {u_down:.1f} m/s")
ax_power_u.plot(u_down, P_down_available, 'o', color='red')
ax_power_u.plot(u_down, P_down, 'x', color='orange', label=fr"$\eta_{{\mathrm{{down}}}}$ = {factor_down:.2f}, $P_{{\mathrm{{down}}}}$ = {P_down:.0f}")
ax_power_u.set_ylabel('Power (in MW)')
ax_power_u.legend(loc='upper right')
ax_cp_u.plot(u, Cp_u, label=r"$C_{\mathrm{P}}(u)$")
ax_cp_u.set_ylabel(r"Power coefficient (in -)")
ax_cp_u.legend(loc='upper right')
ax_ct_u.plot(u, Ct_u, label=r"$C_{\mathrm{T}}(u)$")
ax_ct_u.set_ylabel(r"Thrust coefficient (in -)")
ax_ct_u.legend(loc='upper right')
ax_ct_u.set_xlabel(r"Wind speed $u$ (in m/s)")
ax_ct_u.set_xlim([0, 30])

# Plot the Cp and Ct curves
fig_cp_ct_pitch_tsr, (ax_cp, ax_ct, ax_stall) = plt.subplots(1, 3)
col_cp = ax_cp.imshow(Cp, aspect='auto', cmap='inferno', vmin=0, vmax=np.nanmax(Cp), extent=(*pitch_range, *tsr_range))
ax_cp.imshow(Cp_agrees, aspect='auto', cmap='Greens', vmin=0, vmax=np.nanmax(Cp), extent=(*pitch_range, *tsr_range))
ax_cp.plot(np.full(tsr_lambda_available.size, 0), tsr_lambda_available, 'o', color='red', label=fr"u = {u_down}, $P_{{\mathrm{{available}}}}$")
ax_cp.legend(loc='upper right')
ax_cp_rotor_speed = ax_cp.twinx()
ax_cp_rotor_speed.set_ylabel(fr"Rotor speed for $u_{{\mathrm{{down}}}}$ = {u_down} (in RPM)")
ax_cp_rotor_speed.set_ylim(rotor_speed_down)
col_ct = ax_ct.imshow(Ct, aspect='auto', cmap='inferno', vmin=0, vmax=np.nanmax(Ct), extent=(*pitch_range, *tsr_range))
ax_stall.imshow(stall, aspect='auto', cmap='coolwarm', vmin=0, vmax=1, extent=(*pitch_range, *tsr_range))
ax_cp.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_ct.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_stall.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_cp.set_ylabel(r"Tip speed ratio (TSR) $\lambda$ (in -)")
ax_cp.set_title(r"$C_{\mathrm{P}}(\lambda, \beta)$")
ax_ct.set_title(r"$C_{\mathrm{T}}(\lambda, \beta)$")
plt.colorbar(col_cp, ax=ax_cp, location='top')
plt.colorbar(col_ct, ax=ax_ct, location='top')
fig_cp_ct_pitch_tsr.suptitle('Cp and Ct Curves')

# Show the plots
# plt.close(fig_cp_ct_u)
# plt.close(fig_cp_ct)
plt.show()