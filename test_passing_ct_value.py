"""Script to test overwriting a CT value in a existing FLORIS model"""

from pathlib import Path

import numpy as np
from floris import FlorisModel

# ------ PARAMETERS ------

# Define the wind speeds, directions, and turbulence intensities
wind_speeds = np.array([8.0, 10.0, 12.0, 13.0])  # m/s
wind_directions = np.array([270.0, 270.0, 270.0, 270.0])  # degrees
turbulence_intensities = np.array([0.1, 0.15, 0.2, 0.16])

# Define the yaw angles for the turbines
yaw_setpoints = np.array([[0.0, 10.0, 20.0],
                          [0.0, 5.0, 15.0], 
                          [0.0, 0.0, 10.0], 
                          [0.0, -5.0, -15.0]])  # degrees

# Set the CT value to be passed to the model
ct_value = 0.1

# ------ SCRIPT ------

# Create a FLORIS object
wfm = FlorisModel(Path(__file__).parent / '02_Examples_and_Cases' / '00_Inputs' / '01_FLORIS' / 'gch.yaml')

# Set the wind speed and direction
wfm.set(wind_directions=[wd for wd in wind_directions], wind_speeds=[ws for ws in wind_speeds], turbulence_intensities=[ti for ti in turbulence_intensities])

# Set the yaw angles
wfm.set(yaw_angles=[ys for ys in yaw_setpoints])

# Run the model
wfm.run()

# Extract the powers
power = wfm.get_turbine_powers()

# Get the wind speeds
u = wfm.core.farm.turbine_map[0].power_thrust_table['wind_speed']

# Set the new CT values for the first turbine
wfm.core.farm.turbine_map[0].power_thrust_table['thrust_coefficient'] = np.full(u.size, ct_value)

# Run the model again
wfm.run()

# Extract the new powers
powers_new = wfm.get_turbine_powers()

# ------ PRINTING ------

np.set_printoptions(precision=2, suppress=True)

print(f"Turbine Powers (in kW):\n{power * 1E-3} (shape: {power.shape})")
print(f"\nNew Turbine Powers (in kW):\n{powers_new * 1E-3} (shape: {powers_new.shape})")