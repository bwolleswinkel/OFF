"""Script to find the minimum Ct value"""

from pathlib import Path

import numpy as np
import scipy as sp
from floris import FlorisModel
import matplotlib.pyplot as plt

# ------ PARAMETERS ------

# Select the file to read
file_name: Path = Path('02_Examples_and_Cases') / '00_Inputs' / '00_OFF' / '05_Turbine' / 'NREL5MW' / 'Cp_Ct_NREL5MW_nrel.csv'

# ------ SCRIPT ------

# Load the data
data = np.loadtxt(file_name, delimiter=';', skiprows=1)
(pitch, tsr, Cp, Ct), stall = [np.flipud(data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)
pitch_range, tsr_range = np.unique(pitch), np.unique(tsr)

# Set the rated power
air_density = 1.225  # in kg/m^3
rated_power = 5E6  # in W
rotor_radius = 63  # in m
generator_efficiency = 0.994

# Create an interpolation function for Cp and Ct
Cp_interp = lambda lbd_tsr, lbd_pitch: sp.interpolate.griddata(np.array((tsr.flatten(), pitch.flatten())).T, Cp.flatten(), (lbd_tsr, lbd_pitch))
Ct_interp = lambda lbd_tsr, lbd_pitch: sp.interpolate.griddata(np.array((tsr.flatten(), pitch.flatten())).T, Ct.flatten(), (lbd_tsr, lbd_pitch))

# Interpolate the function
num_int_points = 500
tsr_eval, pitch_eval = np.meshgrid(np.linspace(tsr_range[0], tsr_range[-1], num_int_points), np.linspace(pitch_range[0], pitch_range[-1], num_int_points), indexing='ij')
Cp_eval = Cp_interp(tsr_eval, pitch_eval)
Ct_eval = Ct_interp(tsr_eval, pitch_eval)

# Find the maximum Cp value and the corresponding TSR and pitch
max_cp_idx = np.unravel_index(np.nanargmax(Cp_eval), Cp_eval.shape)
tsr_opt, pitch_opt = tsr_eval[max_cp_idx], pitch_eval[max_cp_idx]
max_cp, max_ct = Cp_eval[max_cp_idx], Ct_interp(tsr_opt, pitch_opt)

# Find the Cp for each combination of power setpoint and wind speed
num_eval_points = 100
P_vals, u_vals = np.linspace(0.01, 1, num_eval_points), np.linspace(0.01, 25, num_eval_points)
Ct_mat, Ct_mat_max, rotor_speed_mat, rotor_speed_mat_max = [np.full((num_eval_points, num_eval_points), np.nan) for _ in range(4)]
for idx_eta, eta in enumerate(P_vals):
    for idx_u, u in enumerate(u_vals):
        # Compute the power setpoint
        P = eta * rated_power
        # Compute the Cp value
        Cp_needed = 2 * P / (air_density * np.pi * (rotor_radius ** 2) * (u ** 3) * generator_efficiency)
        # Find the minimum Ct value for this Cp value
        if Cp_needed > max_cp:
            Ct_mat[idx_eta, idx_u] = max_ct
            Ct_mat_max[idx_eta, idx_u] = max_ct
            # Calculate the rotor speed for this power setpoint and wind speed, blade pitch is zero for maximum Cp
            rotor_speed_mat[idx_eta, idx_u] = tsr_opt * u / rotor_radius * (60 / (2 * np.pi))  # Convert to RPM
            rotor_speed_mat_max[idx_eta, idx_u] = tsr_opt * u / rotor_radius * (60 / (2 * np.pi))  # Convert to RPM
        else:
            # Find all the Cp values in the interpolated matrix that are close to the needed Cp value
            Cp_diff = np.abs(Cp_eval - Cp_needed)
            close_cp_indices = np.where(Cp_diff < 0.01 * Cp_needed)
            if close_cp_indices[0].size > 0:
                best_idx = np.nanargmin(Ct_eval[close_cp_indices])
                best_idx_max = np.nanargmax(Ct_eval[close_cp_indices])
                Ct_mat[idx_eta, idx_u] = Ct_eval[close_cp_indices][best_idx]
                Ct_mat_max[idx_eta, idx_u] = Ct_eval[close_cp_indices][best_idx_max]
                # Calculate the rotor speed for this power setpoint and wind speed, blade pitch is zero for maximum Cp
                rotor_speed_mat[idx_eta, idx_u] = tsr_eval[close_cp_indices][best_idx] * u / rotor_radius * (60 / (2 * np.pi))  # Convert to RPM
                rotor_speed_mat_max[idx_eta, idx_u] = tsr_eval[close_cp_indices][best_idx_max] * u / rotor_radius * (60 / (2 * np.pi))  # Convert to RPM


# TEMP: Save the results
np.save('dr_omega_vals.npy', rotor_speed_mat)
np.save('dr_P_fraction_vals.npy', P_vals)
np.save('dr_u_vals.npy', u_vals)

# ------ PLOTTING ------

# Plot the Cp and Ct curves
fig_cp_ct_pitch_tsr, (ax_cp, ax_ct) = plt.subplots(1, 2)
col_cp = ax_cp.imshow(Cp, aspect='auto', cmap='inferno', vmin=0, vmax=np.nanmax(Cp), extent=(*pitch_range[[0, -1]], *tsr_range[[0, -1]]))
col_ct = ax_ct.imshow(Ct, aspect='auto', cmap='inferno', vmin=0, vmax=np.nanmax(Ct), extent=(*pitch_range[[0, -1]], *tsr_range[[0, -1]]))
ax_cp.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_ct.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_cp.set_ylabel(r"Tip speed ratio (TSR) $\lambda$ (in -)")
ax_cp.set_title(r"$C_{\mathrm{P}}(\lambda, \beta)$")
ax_ct.set_title(r"$C_{\mathrm{T}}(\lambda, \beta)$")
plt.colorbar(col_cp, ax=ax_cp, location='top')
plt.colorbar(col_ct, ax=ax_ct, location='top')
fig_cp_ct_pitch_tsr.suptitle('Cp and Ct Curves')

# Plot the interpolated Cp and Ct curves
fig_cp_ct_interp, (ax_cp_interp, ax_ct_interp) = plt.subplots(1, 2)
col_cp_interp = ax_cp_interp.imshow(Cp_eval, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(Cp_eval), extent=(*pitch_range[[0, -1]], *tsr_range[[0, -1]]))
col_ct_interp = ax_ct_interp.imshow(Ct_eval, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(Ct_eval), extent=(*pitch_range[[0, -1]], *tsr_range[[0, -1]]))
ax_cp_interp.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_ct_interp.set_xlabel(r"Blade pitch $\beta$ (in °)")
ax_cp_interp.set_ylabel(r"Tip speed ratio (TSR) $\lambda$ (in -)")
ax_cp_interp.set_title(r"$C_{\mathrm{P}}(\lambda, \beta)$")
ax_ct_interp.set_title(r"$C_{\mathrm{T}}(\lambda, \beta)$")
plt.colorbar(col_cp_interp, ax=ax_cp_interp, location='top')
plt.colorbar(col_ct_interp, ax=ax_ct_interp, location='top')
fig_cp_ct_interp.suptitle('Interpolated Cp and Ct Curves')

# Plot the minimum Ct values for each power setpoint and wind speed
fig_ct_min, ax_ct_min = plt.subplots()
col_ct_min = ax_ct_min.imshow(Ct_mat, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(Ct_mat), extent=(u_vals[0], u_vals[-1], P_vals[0], P_vals[-1]))
ax_ct_min.set_xlabel(r"Wind speed $u$ (in m/s)")
ax_ct_min.set_ylabel(r"Power setpoint $\eta$ (in -)")
ax_ct_min.set_title(r"Minimum $C_{\mathrm{T}}$ for each power setpoint and wind speed")
plt.colorbar(col_ct_min, ax=ax_ct_min, location='top')

# Plot the rotor speed for each power setpoint and wind speed
fig_rotor_speed, ax_rotor_speed = plt.subplots()
col_rotor_speed = ax_rotor_speed.imshow(rotor_speed_mat, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(rotor_speed_mat), extent=(u_vals[0], u_vals[-1], P_vals[0], P_vals[-1]))
ax_rotor_speed.set_xlabel(r"Wind speed $u$ (in m/s)")
ax_rotor_speed.set_ylabel(r"Power setpoint $\eta$ (in -)")
ax_rotor_speed.set_title(r"Rotor speed for each power setpoint and wind speed (in RPM)")
plt.colorbar(col_rotor_speed, ax=ax_rotor_speed, location='top')

# Plot the maximum Ct values for each power setpoint and wind speed
fig_ct_max, ax_ct_max = plt.subplots()
col_ct_max = ax_ct_max.imshow(Ct_mat_max, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(Ct_mat_max), extent=(u_vals[0], u_vals[-1], P_vals[0], P_vals[-1]))
ax_ct_max.set_xlabel(r"Wind speed $u$ (in m/s)")
ax_ct_max.set_ylabel(r"Power setpoint $\eta$ (in -)")
ax_ct_max.set_title(r"Maximum $C_{\mathrm{T}}$ for each power setpoint and wind speed")
plt.colorbar(col_ct_max, ax=ax_ct_max, location='top')

# Plot the rotor speed for each power setpoint and wind speed for maximum Ct
fig_rotor_speed_max, ax_rotor_speed_max = plt.subplots()
col_rotor_speed_max = ax_rotor_speed_max.imshow(rotor_speed_mat_max, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=np.nanmax(rotor_speed_mat_max), extent=(u_vals[0], u_vals[-1], P_vals[0], P_vals[-1]))
ax_rotor_speed_max.set_xlabel(r"Wind speed $u$ (in m/s)")
ax_rotor_speed_max.set_ylabel(r"Power setpoint $\eta$ (in -)")
ax_rotor_speed_max.set_title(r"Rotor speed for each power setpoint and wind speed for maximum $C_{\mathrm{T}}$ (in RPM)")
plt.colorbar(col_rotor_speed_max, ax=ax_rotor_speed_max, location='top')

# Show the plots
plt.show()