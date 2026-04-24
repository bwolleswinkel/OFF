"""Script to test the evaluation of a fast wake model based on the Bastankhah wake modes"""
# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/25

import numpy as np
from numpy.typing import NDArray
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import multiprocessing as mp
from functools import partial


def calc_vel(coords: NDArray, layout: NDArray, hub_heights: NDArray, rotor_diameters: NDArray, wind_direction: float, wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, apply_shear: bool = False) -> NDArray:
    """Calculated the velocity at the given coordinates based on wake of the turbines"""
    k_star = 0.08  # Random placeholder | Marcus: this is a function of TI
    PARAM_BART = 3  # NOTE: Not there in the equations, but provides a better 'fit'?

    def calc_deficit(rel_coords: NDArray, hub_height: float, rotor_diameter: float, yaw_angle: float, Ct: float) -> NDArray:
        """Calculate the wake deficit at the given coordinates based on the Bastankhah wake model"""

        def delta(x: NDArray, rotor_diameter: float, yaw_angle: float, Ct: float) -> NDArray:
            """Calculate the lateral deflection of the wake centerline based on the yaw angle and turbulence intensity"""
            k_d = 0.01  # Random placeholder

            def xi_init(yaw_angle: float, Ct: float) -> float:
                """Calculate the initial wake deflection based on the yaw angle and turbulence intensity"""
                return 1/2 * np.cos(yaw_angle) ** 2 * np.sin(yaw_angle) * Ct
            
            xi_init_val = xi_init(yaw_angle, Ct)
            delta_val = ((xi_init_val * (15 * ((2 * k_d * x) / (rotor_diameter) + 1) ** 4 + xi_init_val ** 2)) / (((30 * k_d) / (rotor_diameter)) * ((2 * k_d * x) / (rotor_diameter) + 1) ** 5)) - (xi_init_val * rotor_diameter * (15 + xi_init_val ** 2)) / (30 * k_d)
            return delta_val
        
        deficit = np.zeros(rel_coords.shape[0])
        # Create mask for coordinates where x > 0 (downstream of turbine)
        downstream_mask = rel_coords[:, 0] > 0
        # Only calculate deficit for downstream coordinates
        if np.any(downstream_mask):
            beta = (1 / 2) * ((1 + np.sqrt(1 - Ct)) / (np.sqrt(Ct)))
            x_prime, y_prime, z = rel_coords[downstream_mask, 0], rel_coords[downstream_mask, 1], rel_coords[downstream_mask, 2]
            denominator = k_star * (x_prime / rotor_diameter) + 0.2 * np.sqrt(beta) ** 2
            deficit[downstream_mask] = (1 - np.sqrt(1 - (Ct / (8 * denominator)))) * np.exp(- (1 / (2 * denominator)) * (((z - hub_height) / rotor_diameter) ** 2 + PARAM_BART * ((y_prime - delta(x_prime, rotor_diameter, yaw_angle, Ct)) / rotor_diameter) ** 2))
        return deficit
    
    def rot_mat_z(angle: float) -> NDArray:
        """Calculate the rotation matrix around the z-axis, clockwise, with a 90-degrees offset"""
        # NOTE: Assumes the input angle is in rads
        angle_rad = np.radians(np.degrees(angle) + 90)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rot_mat = np.array([[cos_angle, -sin_angle, 0],
                            [sin_angle, cos_angle, 0],
                            [0, 0, 1]])
        return rot_mat

    # Convert input angles to radians
    wind_direction, yaw_angles = np.radians(wind_direction), np.radians(yaw_angles)
    # Initialize the velocity field
    deficits = np.zeros((coords.shape[0], layout.shape[0]))
    # Loop over the turbines
    for idx in range(layout.shape[0]):
        # Calculate the relative coordinates
        rel_coords = (coords - np.concatenate([layout[idx], [0]])) @ rot_mat_z(wind_direction).T
        # Apply horizontal shear to align wake with yawed rotor plane, NOTE: Non-physical
        if apply_shear:
            rel_coords[:, 0] = rel_coords[:, 0] + rel_coords[:, 1] * np.tan(yaw_angles[idx])
        # Calculate the wake deficit at the given coordinates
        deficits[:, idx] = calc_deficit(rel_coords, hub_heights[idx], rotor_diameters[idx], yaw_angles[idx], Ct[idx])
    # Combine the deficits from all turbines using sum of squares of deficits
    # (Delta_U_total / Delta_inf)^2 = sum((Delta_U_i / Delta_inf)^2) for all turbines i
    # where deficits contains Delta_U_i / Delta_inf
    combined_deficit_squared = np.sum(deficits ** 2, axis=1)
    combined_deficit = np.sqrt(combined_deficit_squared)
    # Calculate the velocity at the given coordinates: U_wake = Delta_inf * (1 - Delta_U_total / Delta_inf)
    vels = wind_speed * (1 - combined_deficit)
    return vels


def calc_vel_avg(layout: NDArray, hub_heights: NDArray, rotor_diameters: NDArray, wind_direction: float, wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, num_points: int = 20, apply_shear: bool = False) -> NDArray:
    """Calculate the average velocity at the given coordinates based on wake of the turbines"""
    # Calculate the rotor plane
    vels_rotor_planes = np.zeros((num_points ** 2, layout.shape[0]))
    for idx in range(layout.shape[0]):
        # Extract the turbine position and parameters
        turbine_x = layout[idx, 0]
        turbine_y = layout[idx, 1]
        turbine_z = hub_heights[idx]
        rotor_radius = rotor_diameters[idx] / 2
        # Create a grid of points on the y-z plane, centered at the origin
        y_grid, z_grid = np.meshgrid(np.linspace(-rotor_radius, rotor_radius, num_points), 
                                    np.linspace(-rotor_radius, rotor_radius, num_points), indexing='xy')
        x_grid = np.full_like(y_grid, -10)  # FIXME: Why do I need this slight offset? Seems way to numerically prone | This offset seems to determine everything, something is definitely wrong; the numerical 'filtering' of `calc_vel` is way off. 
        # Rotate the grid to align with the wind direction
        if apply_shear:
            R = rot_mat_z_numba(wind_direction + yaw_angles[idx])  # FIXME: Why is it plus here?
        else:    
            R = rot_mat_z_numba(wind_direction)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]) @ R.T
        # Translate the points to the turbine position
        points += np.array([turbine_x, turbine_y, turbine_z])
        # Calculate the velocity at the rotor plane points
        vels_rotor_planes[:, idx] = calc_vel(points, layout, hub_heights, rotor_diameters,
                                            wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, apply_shear=apply_shear)
    # Calculate the rotor-plane average velocity for each turbine. First, extract only the points that are within the rotor radius in the y-z plane, then average the velocity for those points.
    rotor_plane_avg_vels = np.zeros(layout.shape[0])
    for idx in range(layout.shape[0]):
        # Extract the points for this turbine
        points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        # Calculate the distance from the center of the rotor plane in the y-z plane
        distance_from_center = np.sqrt(points[:, 1]**2 + points[:, 2]**2)
        # Create a mask for points within the rotor radius
        within_rotor_mask = distance_from_center <= rotor_radius
        # Average the velocity for points within the rotor radius
        if np.any(within_rotor_mask):
            rotor_plane_avg_vels[idx] = np.mean(vels_rotor_planes[within_rotor_mask, idx])
        else:
            rotor_plane_avg_vels[idx] = np.nan  # No points within rotor, set to NaN
    return rotor_plane_avg_vels


def calc_ti(coords: NDArray, layout: NDArray, hub_heights: NDArray, rotor_diameters: NDArray, wind_direction: float, wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, apply_shear: bool = False) -> NDArray:
    """Calculate the turbulence intensity at the given coordinates based on wake of the turbines"""
    # NOTE: Based on the implementation/comments of Marcus
    k_star = 0.08  # Random placeholder | Marcus: this is a function of TI
    PARAM_BART = 3  # NOTE: Not there in the equations, but provides a better 'fit'?

    def calc_increment(rel_coords: NDArray, hub_height: float, rotor_diameter: float, turbulence_intensity: float, yaw_angle: float, Ct: float) -> NDArray:
        """Calculate the increment in turbulence intensity at the given coordinates based on Marcus' implementation/comments"""

        def delta(x: NDArray, rotor_diameter: float, yaw_angle: float, Ct: float) -> NDArray:
            """Calculate the lateral deflection of the wake centerline based on the yaw angle and turbulence intensity"""
            k_d = 0.01  # Random placeholder

            def xi_init(yaw_angle: float, Ct: float) -> float:
                """Calculate the initial wake deflection based on the yaw angle and turbulence intensity"""
                return 1/2 * np.cos(yaw_angle) ** 2 * np.sin(yaw_angle) * Ct
            
            xi_init_val = xi_init(yaw_angle, Ct)
            delta_val = ((xi_init_val * (15 * ((2 * k_d * x) / (rotor_diameter) + 1) ** 4 + xi_init_val ** 2)) / (((30 * k_d) / (rotor_diameter)) * ((2 * k_d * x) / (rotor_diameter) + 1) ** 5)) - (xi_init_val * rotor_diameter * (15 + xi_init_val ** 2)) / (30 * k_d)
            return delta_val
        
        def ax_ind(yaw_angle: float, Ct: float) -> float:
            """Calculate the axial induction factor based with a correction for yaw based on the thrust coefficient"""
            # Solve Ct = 4 * a * (1 - a) * cos(gamma) for a
            # Using quadratic formula: a = (1 - sqrt(1 - Ct/cos(gamma))) / 2
            cos_gamma = np.cos(yaw_angle)
            # Ensure Ct/cos(gamma) <= 1 for valid solution
            discriminant = 1 - Ct / cos_gamma
            if discriminant < 0:
                # If discriminant is negative, return limiting case
                # NOTE: Theoretically, this should only happen if yaw_angle >= 45 degrees
                return 0.5  # Maximum theoretical value for axial induction factor
            a_yaw = (1 - np.sqrt(discriminant)) / 2
            return a_yaw
        
        increment = np.zeros(rel_coords.shape[0])
        # Create mask for coordinates where x > 0 (downstream of turbine)
        downstream_mask = rel_coords[:, 0] > 0
        # Only calculate increment for downstream coordinates
        if np.any(downstream_mask):
            beta = (1 / 2) * ((1 + np.sqrt(1 - Ct)) / (np.sqrt(Ct)))
            x_prime, y_prime, z = rel_coords[downstream_mask, 0], rel_coords[downstream_mask, 1], rel_coords[downstream_mask, 2]
            wake = 0.5 * ax_ind(yaw_angle, Ct) ** 0.8 * turbulence_intensity ** 0.1 * (x_prime / rotor_diameter) ** (-0.32)
            denominator = k_star * (x_prime / rotor_diameter) + 0.2 * np.sqrt(beta) ** 2
            increment[downstream_mask] = wake * np.exp(- (1 / (2 * denominator)) * (((z - hub_height) / rotor_diameter) ** 2 + PARAM_BART * ((y_prime - delta(x_prime, rotor_diameter, yaw_angle, Ct)) / rotor_diameter) ** 2))
        return increment
    
    def rot_mat_z(angle: float) -> NDArray:
        """Calculate the rotation matrix around the z-axis, clockwise, with a 90-degrees offset"""
        # NOTE: Assumes the input angle is in rads
        angle_rad = np.radians(np.degrees(angle) + 90)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rot_mat = np.array([[cos_angle, -sin_angle, 0],
                            [sin_angle, cos_angle, 0],
                            [0, 0, 1]])
        return rot_mat

    # Convert input angles to radians
    wind_direction, yaw_angles = np.radians(wind_direction), np.radians(yaw_angles)
    # Initialize the TI field
    increments = np.zeros((coords.shape[0], layout.shape[0]))
    # Loop over the turbines
    for idx in range(layout.shape[0]):
        # Calculate the relative coordinates
        rel_coords = (coords - np.concatenate([layout[idx], [0]])) @ rot_mat_z(wind_direction).T
        # Apply horizontal shear to align wake with yawed rotor plane, NOTE: Non-physical
        if apply_shear:
            rel_coords[:, 0] = rel_coords[:, 0] + rel_coords[:, 1] * np.tan(yaw_angles[idx])
        # Calculate the TI increment at the given coordinates
        increments[:, idx] = calc_increment(rel_coords, hub_heights[idx], rotor_diameters[idx], turbulence_intensity, yaw_angles[idx], Ct[idx])
    # Combine the increments based on the max TI rule
    increment = np.max(np.sqrt(increments ** 2 + turbulence_intensity ** 2), axis=1)
    # Return the results
    return increment


def calc_flow(coords: NDArray, layout: NDArray, hub_heights: NDArray, rotor_diameters: NDArray, wind_direction: float, wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, apply_shear: bool = False) -> tuple[NDArray, NDArray]:
    """Calculate the wind speed and turbulence intensity at the given coordinates based on wake of the turbines"""
    # NOTE: Based on the implementation/comments of Marcus
    k_star = 0.08  # Random placeholder | Marcus: this is a function of TI
    PARAM_BART = 3  # NOTE: Not there in the equations, but provides a better 'fit'?

    def calc_vel_deficit_calc_ti_increment(rel_coords: NDArray, hub_height: float, rotor_diameter: float, turbulence_intensity: float, yaw_angle: float, Ct: float) -> tuple[NDArray, NDArray]:
        """Calculate the wake deficit and TI increment at the given coordinates based on the Bastankhah wake model"""

        def delta(x: NDArray, rotor_diameter: float, yaw_angle: float, Ct: float) -> NDArray:
            """Calculate the lateral deflection of the wake centerline based on the yaw angle and turbulence intensity"""
            k_d = 0.01  # Random placeholder

            def xi_init(yaw_angle: float, Ct: float) -> float:
                """Calculate the initial wake deflection based on the yaw angle and turbulence intensity"""
                return 1/2 * np.cos(yaw_angle) ** 2 * np.sin(yaw_angle) * Ct
            
            xi_init_val = xi_init(yaw_angle, Ct)
            delta_val = ((xi_init_val * (15 * ((2 * k_d * x) / (rotor_diameter) + 1) ** 4 + xi_init_val ** 2)) / (((30 * k_d) / (rotor_diameter)) * ((2 * k_d * x) / (rotor_diameter) + 1) ** 5)) - (xi_init_val * rotor_diameter * (15 + xi_init_val ** 2)) / (30 * k_d)
            return delta_val
        
        def ax_ind(yaw_angle: float, Ct: float) -> float:
            """Calculate the axial induction factor based with a correction for yaw based on the thrust coefficient"""
            # Solve Ct = 4 * a * (1 - a) * cos(gamma) for a
            # Using quadratic formula: a = (1 - sqrt(1 - Ct/cos(gamma))) / 2
            cos_gamma = np.cos(yaw_angle)
            # Ensure Ct/cos(gamma) <= 1 for valid solution
            discriminant = 1 - Ct / cos_gamma
            if discriminant < 0:
                # If discriminant is negative, return limiting case
                # NOTE: Theoretically, this should only happen if yaw_angle >= 45 degrees
                return 0.5  # Maximum theoretical value for axial induction factor
            a_yaw = (1 - np.sqrt(discriminant)) / 2
            return a_yaw
        
        deficit, increment = np.zeros(rel_coords.shape[0]), np.zeros(rel_coords.shape[0])
        # Create mask for coordinates where x > 0 (downstream of turbine)
        downstream_mask = rel_coords[:, 0] > 0
        # Only calculate deficit for downstream coordinates
        if np.any(downstream_mask):
            beta = (1 / 2) * ((1 + np.sqrt(1 - Ct)) / (np.sqrt(Ct)))
            x_prime, y_prime, z = rel_coords[downstream_mask, 0], rel_coords[downstream_mask, 1], rel_coords[downstream_mask, 2]
            wake = 0.5 * ax_ind(yaw_angle, Ct) ** 0.8 * turbulence_intensity ** 0.1 * (x_prime / rotor_diameter) ** (-0.32)
            denominator = k_star * (x_prime / rotor_diameter) + 0.2 * np.sqrt(beta) ** 2
            exp_factor = np.exp(- (1 / (2 * denominator)) * (((z - hub_height) / rotor_diameter) ** 2 + PARAM_BART * ((y_prime - delta(x_prime, rotor_diameter, yaw_angle, Ct)) / rotor_diameter) ** 2))
            deficit[downstream_mask] = (1 - np.sqrt(1 - (Ct / (8 * denominator)))) * exp_factor
            increment[downstream_mask] = wake * exp_factor
        return deficit, increment
    
    def rot_mat_z(angle: float) -> NDArray:
        """Calculate the rotation matrix around the z-axis, clockwise, with a 90-degrees offset"""
        # NOTE: Assumes the input angle is in rads
        angle_rad = np.radians(np.degrees(angle) + 90)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rot_mat = np.array([[cos_angle, -sin_angle, 0],
                            [sin_angle, cos_angle, 0],
                            [0, 0, 1]])
        return rot_mat

    # Convert input angles to radians
    wind_direction, yaw_angles = np.radians(wind_direction), np.radians(yaw_angles)
    # Initialize the velocity field and TI
    vel_defecits, ti_increments = np.zeros((coords.shape[0], layout.shape[0])), np.zeros((coords.shape[0], layout.shape[0]))
    # Loop over the turbines
    for idx in range(layout.shape[0]):
        # Calculate the relative coordinates
        rel_coords = (coords - np.concatenate([layout[idx], [0]])) @ rot_mat_z(wind_direction).T
        # Apply horizontal shear to align wake with yawed rotor plane, NOTE: Non-physical
        if apply_shear:
            rel_coords[:, 0] = rel_coords[:, 0] + rel_coords[:, 1] * np.tan(yaw_angles[idx])
        # Calculate the velocity deficit and TI increment at the given coordinates
        vel_defecits[:, idx], ti_increments[:, idx] = calc_vel_deficit_calc_ti_increment(rel_coords, hub_heights[idx], rotor_diameters[idx], turbulence_intensity, yaw_angles[idx], Ct[idx])
    # Combine the deficits from all turbines using sum of squares of deficits
    combined_deficit_squared = np.sum(vel_defecits ** 2, axis=1)
    combined_deficit = np.sqrt(combined_deficit_squared)
    vel = wind_speed * (1 - combined_deficit)
    # Combine the increments based on the max TI rule
    increment = np.max(np.sqrt(ti_increments ** 2 + turbulence_intensity ** 2), axis=1)
    # Return the results
    return vel, increment


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/25
# Numba-optimized versions for performance
@nb.njit
def rot_mat_z_numba(angle: float):
    """Calculate the rotation matrix around the z-axis, clockwise, with a 90-degrees offset - Numba version"""
    angle_rad = np.radians(angle + 90)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rot_mat = np.array([[cos_angle, -sin_angle, 0.0],
                        [sin_angle, cos_angle, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_mat


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/25
@nb.njit
def calc_vel_numba(coords: NDArray, layout: NDArray, hub_heights: NDArray, 
                   rotor_diameters: NDArray, wind_direction: float, 
                   wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray) -> NDArray:
    """Calculated the velocity at the given coordinates based on wake of the turbines - Numba version"""
    
    # Pre-calculate rotation matrix
    rot_mat = rot_mat_z_numba(wind_direction)
    
    # Initialize the velocity field
    deficits = np.zeros((coords.shape[0], layout.shape[0]), dtype=np.float64)
    
    # Loop over the turbines
    for idx in range(layout.shape[0]):
        # Calculate the relative coordinates
        for i in range(coords.shape[0]):
            # Translate coordinates relative to turbine
            rel_x = coords[i, 0] - layout[idx, 0]
            rel_y = coords[i, 1] - layout[idx, 1] 
            rel_z = coords[i, 2] - 0.0
            
            # Apply rotation matrix manually (more numba-friendly)
            new_x = rot_mat[0, 0] * rel_x + rot_mat[0, 1] * rel_y + rot_mat[0, 2] * rel_z
            new_y = rot_mat[1, 0] * rel_x + rot_mat[1, 1] * rel_y + rot_mat[1, 2] * rel_z
            new_z = rot_mat[2, 0] * rel_x + rot_mat[2, 1] * rel_y + rot_mat[2, 2] * rel_z
            
            # Calculate deficit for this point and turbine
            if new_x > 0:  # downstream check
                k_star = 0.08
                beta = (1 / 2) * ((1 + np.sqrt(1 - Ct[idx])) / (np.sqrt(Ct[idx])))
                denominator = k_star * (new_x / rotor_diameters[idx]) + 0.2 * np.sqrt(beta) ** 2
                
                # Calculate exponential term
                z_term = (new_z - hub_heights[idx]) / rotor_diameters[idx]
                y_term = new_y / rotor_diameters[idx]
                exp_term = np.exp(-(1 / (2 * denominator)) * (z_term**2 + y_term**2))
                
                deficits[i, idx] = (1 - np.sqrt(1 - (Ct[idx] / (8 * denominator)))) * exp_term
    
    # Combine the deficits from all turbines using sum of squares of deficits
    combined_deficit_squared = np.zeros(coords.shape[0], dtype=np.float64)
    for i in range(coords.shape[0]):
        sum_squared = 0.0
        for j in range(layout.shape[0]):
            deficit = deficits[i, j]
            sum_squared += deficit * deficit
        combined_deficit_squared[i] = sum_squared
    
    combined_deficit = np.sqrt(combined_deficit_squared)
    
    # Calculate the velocity at the given coordinates: U_wake = Delta_inf * (1 - Delta_U_total / Delta_inf)
    vels = wind_speed * (1 - combined_deficit)
    return vels


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/25
# Multiprocessing helper functions
def calc_vel_chunk(coords_chunk: NDArray, layout: NDArray, hub_heights: NDArray, 
                   rotor_diameters: NDArray, wind_direction: float, 
                   wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, use_numba: bool = True) -> NDArray:
    """Process a chunk of coordinates"""
    if use_numba:
        return calc_vel_numba(coords_chunk, layout, hub_heights, rotor_diameters, 
                             wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct)
    else:
        return calc_vel(coords_chunk, layout, hub_heights, rotor_diameters, 
                       wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct)  # turbulence_intensity not used


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/25
def calc_vel_multiprocess(coords: NDArray, layout: NDArray, hub_heights: NDArray, 
                         rotor_diameters: NDArray, wind_direction: float, 
                         wind_speed: float, turbulence_intensity: float, yaw_angles: NDArray, Ct: NDArray, 
                         n_processes: int = None, chunk_size: int = None,
                         use_numba: bool = True) -> NDArray:
    """Calculate velocities using multiprocessing for large coordinate arrays"""
    
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one core free
    
    if chunk_size is None:
        # Automatic chunk sizing based on number of processes
        chunk_size = max(1000, coords.shape[0] // (n_processes * 4))
    
    # Split coordinates into chunks
    n_coords = coords.shape[0]
    chunks = []
    for i in range(0, n_coords, chunk_size):
        end_idx = min(i + chunk_size, n_coords)
        chunks.append(coords[i:end_idx])
    
    # Prepare the function with fixed arguments
    worker_func = partial(calc_vel_chunk, 
                         layout=layout, hub_heights=hub_heights,
                         rotor_diameters=rotor_diameters, wind_direction=wind_direction,
                         wind_speed=wind_speed, turbulence_intensity=turbulence_intensity, yaw_angles=yaw_angles, Ct=Ct, use_numba=use_numba)
    
    # Process chunks in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(worker_func, chunks)
    
    # Concatenate results
    return np.concatenate(results)


def main():
    # ==== RUNTIME PARAMETERS ====
    num_points_topdown = 1_000
    num_points_rotor_plane = 20
    num_wind_directions = 36
    # ==== RUNTIME PARAMETERS ====

    # Set whether or not to apply shear for the wake
    apply_shear = True

    # Wind farm
    layout = np.array([[608, 500],
                       [1500, 500],
                       [2392, 500]])
    
    hub_heights = np.array([120, 120, 120])
    rotor_diameters = np.array([90, 90, 90])

    # Ambient conditions
    wind_direction = 270
    wind_speed = 8.0
    turbulence_intensity = 0.16

    # Placeholder
    Ct = np.array([0.6, 0.6, 0.6])
    yaw_angles = np.array([20.0, 0.0, 0.0])

    # Sample points
    borders_x, borders_y = (0, 3500), (0, 1000)
    x, y = np.meshgrid(np.linspace(*borders_x, num_points_topdown), np.linspace(*borders_y, num_points_topdown), indexing='xy')
    z = np.full_like(x, 120)
    coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Calculate the top-down plane 
    vels_original = calc_vel(coords, layout, hub_heights, rotor_diameters, 
                             wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, apply_shear=apply_shear)
    
    # Calculate the TI for the top-down plane
    ti_original = calc_ti(coords, layout, hub_heights, rotor_diameters, 
                          wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, apply_shear=apply_shear)
    
    # Calculate the flow (velocity and TI) for the top-down plane
    vels_original, ti_original = calc_flow(coords, layout, hub_heights, rotor_diameters, 
                                           wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, apply_shear=apply_shear)
 
    # Calculate the rotor plane
    vels_rotor_planes = np.zeros((num_points_rotor_plane ** 2, layout.shape[0]))
    for idx in range(layout.shape[0]):
        # Extract the turbine position and parameters
        turbine_x = layout[idx, 0]
        turbine_y = layout[idx, 1]
        turbine_z = hub_heights[idx]
        rotor_radius = rotor_diameters[idx] / 2
        # Create a grid of points on the y-z plane, centered at the origin
        y_grid, z_grid = np.meshgrid(np.linspace(-rotor_radius, rotor_radius, num_points_rotor_plane), 
                                     np.linspace(-rotor_radius, rotor_radius, num_points_rotor_plane), indexing='xy')
        x_grid = np.full_like(y_grid, -10)  # FIXME: Why do I need this slight offset? Seems way to numerically prone | This offset seems to determine everything, something is definitely wrong; the numerical 'filtering' of `calc_vel` is way off. 
        # Rotate the grid to align with the wind direction
        if apply_shear:
            R = rot_mat_z_numba(wind_direction + yaw_angles[idx])  # FIXME: Why is it plus here?
        else:    
            R = rot_mat_z_numba(wind_direction)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]) @ R.T
        # Translate the points to the turbine position
        points += np.array([turbine_x, turbine_y, turbine_z])
        # Calculate the velocity at the rotor plane points
        vels_rotor_planes[:, idx] = calc_vel(points, layout, hub_heights, rotor_diameters,
                                             wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, apply_shear=apply_shear)
        
    rotor_plane_avg_vels = calc_vel_avg(layout, hub_heights, rotor_diameters, wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, num_points=num_points_rotor_plane, apply_shear=apply_shear)

    # ==== COPILOT ====

    def run_performance_tests():
    
        print(f"Processing {coords.shape[0]:,} coordinate points with {layout.shape[0]} turbines")
        print(f"Coordinate array shape: {coords.shape}")
        print(f"Layout array shape: {layout.shape}")
        
        # Performance comparison
        results = {}
        
        # Test original implementation
        print("\n1. Testing original implementation...")
        start_time = time.time()
        vels_original = calc_vel(coords, layout, hub_heights, rotor_diameters, 
                                wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct)
        original_time = time.time() - start_time
        results['original'] = (vels_original, original_time)
        print(f"   Original time: {original_time:.4f} seconds")
        
        # Test numba implementation (first run includes compilation time)
        print("\n2. Testing numba implementation (first run - includes compilation)...")
        start_time = time.time()
        vels_numba = calc_vel_numba(coords, layout, hub_heights, rotor_diameters, 
                                    wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct)
        numba_time_first = time.time() - start_time
        print(f"   Numba time (first run): {numba_time_first:.4f} seconds")
        
        # Test numba implementation (second run - compiled)
        print("\n3. Testing numba implementation (compiled)...")
        start_time = time.time()
        vels_numba = calc_vel_numba(coords, layout, hub_heights, rotor_diameters, 
                                    wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct)
        numba_time = time.time() - start_time
        results['numba'] = (vels_numba, numba_time)
        print(f"   Numba time (compiled): {numba_time:.4f} seconds")
        
        # Test multiprocessing with numba (only if enough points to justify overhead)
        if coords.shape[0] > 10000:
            print("\n4. Testing multiprocessing with numba...")
            start_time = time.time()
            vels_mp = calc_vel_multiprocess(coords, layout, hub_heights, rotor_diameters, 
                                        wind_direction, wind_speed, turbulence_intensity, yaw_angles, Ct, use_numba=True)
            mp_time = time.time() - start_time
            results['multiprocessing'] = (vels_mp, mp_time)
            print(f"   Multiprocessing time: {mp_time:.4f} seconds")
        else:
            print("\n4. Skipping multiprocessing test (not enough points to justify overhead)")
        
        # Performance summary
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY:")
        print(f"{'='*60}")
        print(f"Original:              {original_time:.4f}s (baseline)")
        print(f"Numba (compiled):      {numba_time:.4f}s ({original_time/numba_time:.1f}x speedup)")
        if 'multiprocessing' in results:
            mp_time = results['multiprocessing'][1]
            print(f"Multiprocessing+Numba: {mp_time:.4f}s ({original_time/mp_time:.1f}x speedup)")
        
        # Verify results are consistent
        print(f"\n{'='*60}")
        print("RESULT VERIFICATION:")
        print(f"{'='*60}")
        max_diff_numba = np.max(np.abs(vels_original - results['numba'][0]))
        print(f"Max difference (Original vs Numba): {max_diff_numba:.2e}")
        
        if 'multiprocessing' in results:
            max_diff_mp = np.max(np.abs(vels_original - results['multiprocessing'][0]))
            print(f"Max difference (Original vs MP):    {max_diff_mp:.2e}")

    run_performance_tests()

    # ==== COPILOT ====

    # Create a loop over different wind directions
    wind_directions = np.linspace(0, 360, num_wind_directions, endpoint=False)
    time_start = time.time()
    for wd in wind_directions:
        # Calculate the velocity field for the current wind direction
        vels = calc_vel_numba(coords, layout, hub_heights, rotor_diameters, 
                              wd, wind_speed, turbulence_intensity, yaw_angles, Ct)
    time_end = time.time()
    print(f"\nCalculated velocity fields for {len(wind_directions)} wind directions ({len(wind_directions) * num_points_topdown ** 2:_} coords) in {time_end - time_start:.2f} seconds")

    # Create a loop over different wind direction, using multiprocessing on the outer loop
    time_start = time.time()
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        # Use partial to create a function with fixed arguments except wind_direction
        worker_func = partial(calc_vel_numba, coords, layout, hub_heights, rotor_diameters, 
                             wind_speed=wind_speed, turbulence_intensity=turbulence_intensity, 
                             yaw_angles=yaw_angles, Ct=Ct)
        pool.map(worker_func, wind_directions)
    time_end = time.time()
    print(f"\nCalculated velocity fields for {len(wind_directions)} wind directions ({len(wind_directions) * num_points_topdown ** 2:_} coords) with multiprocessing in {time_end - time_start:.2f} seconds")
    
    # Printing
    print(f"Rotor plane average velocities for each turbine:")
    for idx in range(layout.shape[0]):
        print(f"  Turbine {idx+1}: {rotor_plane_avg_vels[idx]:.2f} m/s")

    # Plot velocity field for the top-down plane
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(vels_original.reshape(x.shape), extent=(*borders_x, *borders_y),
                   origin='lower', cmap='inferno')
    
    # Plot turbulence intensity for the top-down plane
    fig_ti, ax_ti = plt.subplots(figsize=(12, 8))
    im_ti = ax_ti.imshow(ti_original.reshape(x.shape), extent=(*borders_x, *borders_y),
                         origin='lower', cmap='inferno')
    
    # Generate the rotor plane (line) for each turbine
    for idx in range(layout.shape[0]):
        turbine_x = layout[idx, 0]
        turbine_y = layout[idx, 1]
        rotor_radius = rotor_diameters[idx] / 2
        
        # Convert wind direction to rotor orientation, incorporating yaw angle
        # Wind coordinate: 0° = top (negative y), clockwise
        # Rotor plane is perpendicular to wind direction
        # Yaw angle is counter-clockwise positive deviation from wind-facing direction
        # Convert to standard math coordinates and add 90° for perpendicular
        rotor_angle_deg = (180 - wind_direction + yaw_angles[idx]) % 360
        rotor_angle = np.radians(rotor_angle_deg)
        
        # Calculate line endpoints (from one blade tip to the other)
        dx = rotor_radius * np.cos(rotor_angle)
        dy = rotor_radius * np.sin(rotor_angle)
        
        line_x = [turbine_x - dx, turbine_x + dx]
        line_y = [turbine_y - dy, turbine_y + dy]
        
        ax.plot(line_x, line_y, c='white', linewidth=1.5)
    
    ax.set_xlabel(fr"$x$ (in m)")
    ax.set_ylabel(f"$y$ (in m)")
    
    # Create colorbar with same height as the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(fr"$u$ (in m/s)")
    plt.tight_layout()

    # Create colorbar with same height as the plot (for TI)
    divider = make_axes_locatable(ax_ti)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im_ti, cax=cax)
    cbar.set_label(fr"$TI$ (in -)")
    plt.tight_layout()
    
    # Show th plots
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=layout.shape[0], figsize=(15, 5))
    if layout.shape[0] == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Calculate shared colorbar range
    vmin_shared = np.min(vels_rotor_planes)
    vmax_shared = np.max(vels_rotor_planes)
    
    for idx in range(layout.shape[0]):
        rotor_radius = rotor_diameters[idx] / 2
        im = axes[idx].imshow(vels_rotor_planes[:, idx].reshape(num_points_rotor_plane, num_points_rotor_plane),
                            extent=(-rotor_radius, rotor_radius, -rotor_radius, rotor_radius),
                            origin='lower', cmap='inferno', vmin=vmin_shared, vmax=vmax_shared)
        # Plot the rotor as a circle in the plot
        circle = plt.Circle((0, 0), rotor_radius, color='white', fill=False, linewidth=1.5)
        axes[idx].add_patch(circle)
        axes[idx].set_title(f"Turbine {idx+1}")
        axes[idx].set_xlabel("Rotor plane $y'$ (in m)")
        axes[idx].set_ylabel("Rotor plane $z$ (in m)")
        # Add colorbar for each subplot
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Velocity $u$ (in m/s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()