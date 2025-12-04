"""Script to test the rotation matrices and translations"""

import numpy as np
import matplotlib.pyplot as plt

# ------ PARAMETERS ------

# Set the coordinates of the turbines
layout = np.array([[500, 500], [100, 800]])

# Set the hub height of the turbine
hub_height = 90
blade_length = 126 / 2

# Set the wind direction in degrees
wind_direction = 270

# Set the yaw angle in degrees
yaw_angle = np.array([0, 20])  # Yaw angles for each turbine

# Set the azimuth angle in degrees
azimuth_angle: callable = lambda wt_idx, t: (t * 10 + wt_idx * 100) % 360  # 10 degrees per second

# Set the time instant
time = -1

# ------ FUNCTIONS ------

def get_blade_coords(t: float, wt_idx: int, blade_idx: int, pos: float) -> np.ndarray:
    """Get the coordinates of the blade points in the global frame.

    Parameters
    ----------
    t : float
        Time instant
    wt_idx : int
        Wind turbine index
    blade_idx : int
        Blade index
    pos : float
        Position along the blade, where pos ∈ (0, 1)

    Returns
    -------
    np.ndarray
        Coordinates of the blade points in the global frame.
    """

    def rot_mat_y(angle_rad: float) -> np.ndarray:
        """Clockwise rotation matrix around the y-axis."""
        # FROM: GitHub Copilot GPT-4o | 2025/12/01
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    
    def rot_mat_z(angle_rad: float) -> np.ndarray:
        """Clockwise rotation matrix around the z-axis."""
        # FROM: GitHub Copilot GPT-4o | 2025/12/01
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    # FIXME: Note that this function assumes several variables exist in the outer scope: `wind_direction`, `yaw_angle`, `azimuth_angle`, `blade_length`
    # Calculate local blade coordinates around the y-axis
    psi_offset = blade_idx * 120
    # NOTE: It seems to be that, within FLORIS, the yaw angles is NOT defined as being positive clockwise; as such, we need to subtract that angle here in order to make the right calculation
    # FROM: https://nrel.github.io/floris/examples/examples_control_optimization/001_opt_yaw_single_ws.html  # nopep8
    azimuth_rad = np.radians(azimuth_angle(wt_idx, t) + psi_offset)
    x_local = rot_mat_y(azimuth_rad) @ np.array([0, 0, pos * blade_length])
    # Calculate the total rotation around the z-axis
    # NOTE: In this model, we are ignoring the tilt angle
    total_yaw_rad = np.radians(wind_direction - yaw_angle[wt_idx])
    x_global = rot_mat_z(total_yaw_rad) @ x_local
    return x_global

# ------ PLOTTING ------

# Create a 3D wind farm figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the wind turbine towers
for wt_idx, (x, y) in enumerate(layout):
    ax.plot([x, x], [y, y], [0, hub_height], color='black', linewidth=2)
    # Plot the blades
    for blade_idx in range(3):
        blade_coords = np.array([get_blade_coords(time, wt_idx, blade_idx, pos) for pos in [0, 1]])
        ax.plot(blade_coords[:, 0] + x, blade_coords[:, 1] + y, blade_coords[:, 2] + hub_height, color='red')
ax.set_xlabel(f"$x$ (in m)")
ax.set_ylabel(f"$y$ (in m)")
ax.set_zlabel(f"$z$ (in m)")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 200)
ax.set_aspect('equal')
ax.view_init(elev=20, azim=-125, roll=0)

# Show the plot
plt.show()