import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d

# --- 1. Turbine Parameters (from wes-10-987-2025.pdf and general assumptions) ---
# For IEA 10 MW turbine, based on Table 1 
D = 198.0  # Rotor diameter [m]
R = D / 2.0  # Rotor radius [m]
Ar = np.pi * R**2  # Rotor area [m^2]
J = 160342052.0  # LSS-equivalent inertia [kg m^2]
rated_power_gen = 10.0e6  # Rated generator power [W]
rated_wind_speed = 11
# Assumed parameters
rho = 1.225  # Air density [kg/m^3] (from 1turbine.yaml) 
eta_g = 0.95  # Generator efficiency (assumed)
beta_pitch = 0.0  # Constant fine pitch angle [degrees] (assumed based on paper's simplification) 
lambda_opt = 8.0  # Optimal tip-speed ratio (typical value for large turbines)
Cp_opt = 0.48  # Optimal power coefficient (typical value, for generic Cp curve)
# --- 2. Simulation Settings (from 1turbine.yaml)  ---
time_step = 4.0  # Simulation time step [s]
time_start = 0.0  # Simulation start time [s]
time_end = 1200.0  # Simulation end time [s]
# Wind speed profile (time-varying) 
wind_speeds_t = np.array([0.0, 600.0, 1200.0])
wind_speeds = np.array([8.0, 12.0, 12.0])
wind_speed_interp = interp1d(wind_speeds_t, wind_speeds, kind='linear')
# --- 3. Initial Conditions ---
initial_wind_speed = wind_speed_interp(time_start)
initial_omega_r = lambda_opt * initial_wind_speed / R  # Initial rotor speed based on optimal TSR
# --- 4. Define Turbine Models ---
Cp_tb_values = np.array([
    [0.0331, 0.0400, 0.0467, 0.0534, 0.0600, 0.0666, 0.0734, 0.0802, 0.0872, 0.0942, 0.1012],
    [0.0466, 0.0546, 0.0626, 0.0705, 0.0785, 0.0866, 0.0949, 0.1034, 0.1119, 0.1202, 0.1278],
    [0.0621, 0.0716, 0.0810, 0.0904, 0.1001, 0.1100, 0.1202, 0.1304, 0.1402, 0.1488, 0.1551],
    [0.0798, 0.0907, 0.1018, 0.1132, 0.1250, 0.1372, 0.1494, 0.1608, 0.1170, 0.1770, 0.1806],
    [0.0994, 0.1123, 0.1257, 0.1395, 0.1539, 0.1682, 0.1815, 0.1924, 0.1995, 0.2031, 0.2038],
    [0.1215, 0.1370, 0.1530, 0.1697, 0.1865, 0.2020, 0.2145, 0.2226, 0.2263, 0.2268, 0.2249],
    [0.1466, 0.1650, 0.1842, 0.2037, 0.2219, 0.2366, 0.2459, 0.2501, 0.2507, 0.2483, 0.2436],
    [0.1749, 0.1968, 0.2191, 0.2404, 0.2579, 0.2691, 0.2743, 0.2751, 0.2725, 0.2674, 0.2603],
    [0.2067, 0.2321, 0.2569, 0.2779, 0.2918, 0.2985, 0.2999, 0.2975, 0.2921, 0.2846, 0.2754],
    [0.2416, 0.2700, 0.2954, 0.3130, 0.3219, 0.3245, 0.3226, 0.3173, 0.3097, 0.3002, 0.2892],
    [0.2797, 0.3097, 0.3321, 0.3443, 0.3486, 0.3476, 0.3428, 0.3538, 0.3259, 0.3146, 0.3018],
    [0.3195, 0.3479, 0.3645, 0.3713, 0.3719, 0.3679, 0.3610, 0.3519, 0.3407, 0.3279, 0.3133],
    [0.3588, 0.3814, 0.3919, 0.3947, 0.3921, 0.3862, 0.3777, 0.3670, 0.3544, 0.3400, 0.3237],
    [0.3936, 0.4091, 0.4149, 0.4144, 0.4100, 0.4027, 0.3930, 0.3810, 0.3670, 0.3511, 0.3331],
    [0.4217, 0.4314, 0.4339, 0.4315, 0.4259, 0.4177, 0.4069, 0.3938, 0.3786, 0.3612, 0.3416],
    [0.4429, 0.4489, 0.4492, 0.4462, 0.4400, 0.4311, 0.4195, 0.4056, 0.3892, 0.3704, 0.3494],
    [0.4576, 0.4618, 0.4618, 0.4586, 0.4524, 0.4430, 0.4310, 0.4163, 0.3989, 0.3789, 0.3563],
    [0.4661, 0.4706, 0.4715, 0.4688, 0.4628, 0.4535, 0.4412, 0.4260, 0.4078, 0.3866, 0.3626],
    [0.4686, 0.4756, 0.4782, 0.4767, 0.4714, 0.4627, 0.4505, 0.4349, 0.4159, 0.3936, 0.3682],
    [0.4649, 0.4759, 0.4816, 0.4822, 0.4784, 0.4705, 0.4587, 0.4429, 0.4233, 0.4001, 0.3733],
    [0.4568, 0.4714, 0.4816, 0.4854, 0.4837, 0.4771, 0.4660, 0.4502, 0.4301, 0.4060, 0.3779],
    [0.4472, 0.4633, 0.4778, 0.4861, 0.4873, 0.4824, 0.4723, 0.4567, 0.4364, 0.4113, 0.3821],
    [0.4372, 0.4539, 0.4707, 0.4839, 0.4889, 0.4864, 0.4775, 0.4625, 0.4420, 0.4162, 0.3856],
    [0.4265, 0.4442, 0.4618, 0.4786, 0.4884, 0.4889, 0.4816, 0.4674, 0.4469, 0.4204, 0.3886],
    [0.4151, 0.4340, 0.4525, 0.4710, 0.4855, 0.4899, 0.4847, 0.4716, 0.4511, 0.4241, 0.3911],
    [0.4031, 0.4233, 0.4430, 0.4624, 0.4802, 0.4891, 0.4866, 0.4749, 0.4547, 0.4272, 0.3930],
    [0.3903, 0.4122, 0.4331, 0.4535, 0.4732, 0.4866, 0.4875, 0.4774, 0.4577, 0.4297, 0.3945],
    [0.3767, 0.4003, 0.4228, 0.4443, 0.4651, 0.4823, 0.4872, 0.4790, 0.4600, 0.4317, 0.3955],
    [0.3622, 0.3879, 0.4120, 0.4348, 0.4567, 0.4763, 0.4855, 0.4798, 0.4617, 0.4333, 0.3962]
])
Cp_tb_tsr = np.array([3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25,
    6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00])
Cp_tb_bpa = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
cp_interp_2d = RectBivariateSpline(Cp_tb_tsr, Cp_tb_bpa, Cp_tb_values)
Cp_opt_for_K_calculation = np.max(Cp_tb_values[:, 0])
rated_rotor_speed = lambda_opt * rated_wind_speed / R # [rad/s]
# --- 4. Define Turbine Models ---
def Cp_function(lambda_tsr, beta_pitch_deg):
    # Ensure lambda_tsr and beta_pitch_deg are within interpolation bounds
    # For robust simulation, consider clipping inputs or handling extrapolation gracefully
    lambda_tsr_clipped = np.clip(lambda_tsr, Cp_tb_tsr.min(), Cp_tb_tsr.max())
    beta_pitch_deg_clipped = np.clip(beta_pitch_deg, Cp_tb_bpa.min(), Cp_tb_bpa.max())
    Cp = cp_interp_2d(lambda_tsr_clipped, beta_pitch_deg_clipped)[0, 0] # [0,0] because RectBivariateSpline returns 2D array
    return np.clip(Cp, 0.0, 0.55) # Still clip Cp to realistic values
def calculate_generator_power(rated_rotor_speed, omega_r, rated_power, k_const):
    """
    Simplified K*omega_r^3 controller, capped at rated power.
    """
    Pg = k_const * omega_r**3
    return Pg
# Start with small Kp, Ki and no Kd, then increase gradually until response is good.
Kp_pitch = 0.003  # Proportional gain for pitch controller (Tune this!)
Ki_pitch = 0.0003 # Integral gain for pitch controller (Tune this!)
max_pitch_rate = np.deg2rad(8) # Max pitch rate [rad/s] (e.g., 8 degrees/second)
min_pitch_angle = np.deg2rad(0) # Minimum pitch angle [rad] (e.g., 0 degrees)
max_pitch_angle = np.deg2rad(30) # Maximum pitch angle [rad] (e.g., 30 degrees)
K_controller = 0.5 * rho * Ar * Cp_opt_for_K_calculation * eta_g * (R / lambda_opt)**3
# --- 5. Simulation Setup ---
num_steps = int((time_end - time_start) / time_step)
time_points = np.linspace(time_start, time_end, num_steps + 1)
# --- 6. Initial Conditions & List Initialization ---
# Calculate initial values for all variables at time_points[0]
current_omega_r = lambda_opt * wind_speed_interp(time_points[0]) / R # This will be the current_omega_r for the first step
initial_U = wind_speed_interp(time_points[0])
initial_pitch_angle = min_pitch_angle # Start with minimum pitch
current_pitch_angle = initial_pitch_angle
integral_error_pitch = 0.0 # Initialize integral error for PI controller
initial_lambda_tsr = (current_omega_r * R) / initial_U
initial_Cp = Cp_function(initial_lambda_tsr,np.rad2deg(min_pitch_angle))
initial_Pr = 0.5 * rho * Ar * initial_Cp * initial_U**3
initial_Pg = calculate_generator_power(initial_U,current_omega_r, rated_power_gen, K_controller)
initial_power_turb = initial_Pg * eta_g
# Initialize lists with initial values (corresponding to time_points[0])
rotor_speeds = [current_omega_r]
generator_powers = [initial_Pg]
aerodynamic_powers = [initial_Pr]
rews_values = [initial_U]
power_turb = [initial_power_turb]
pitch_angles = [initial_pitch_angle]
# --- 7. Simulation Loop (Euler Integration) ---
for i in range(num_steps):
    current_time_step = time_points[i]
    U_current = wind_speed_interp(current_time_step)
    if current_omega_r >= rated_rotor_speed: 
        error_omega_r = current_omega_r - rated_rotor_speed
        # Proportional term
        P_term = Kp_pitch * error_omega_r
        # Integral term (with basic anti-windup)
        integral_error_pitch += error_omega_r * time_step
        # Anti-windup: stop integrating if pitch is at limit and error would drive it further
        if (current_pitch_angle >= max_pitch_angle and error_omega_r > 0) or (current_pitch_angle <= min_pitch_angle and error_omega_r < 0):
            integral_error_pitch -= error_omega_r * time_step 
        I_term = Ki_pitch * integral_error_pitch
        desired_pitch_rate = P_term + I_term
        desired_pitch_rate = np.clip(desired_pitch_rate, -max_pitch_rate, max_pitch_rate)# Limit pitch rate
        new_pitch_angle = current_pitch_angle + desired_pitch_rate * time_step# Update pitch angle
        new_pitch_angle = np.clip(new_pitch_angle, min_pitch_angle, max_pitch_angle)# Limit pitch angle
        current_pitch_angle = new_pitch_angle # Update pitch for next step
    else:
        # --- Region 2: K*omega_r^3 Control (Fixed Pitch) ---
        current_pitch_angle = min_pitch_angle # Keep pitch at optimal (usually 0 degrees or minimum)
    # Calculate Pr and Pg based on the rotor speed and current pitch *at the start of this time step*
    lambda_tsr_current = (current_omega_r * R) / U_current
    Cp_current = Cp_function(lambda_tsr_current, np.rad2deg(current_pitch_angle))
    Pr_current = 0.5 * rho * Ar * Cp_current * U_current**3
    Pg_current = calculate_generator_power(U_current,current_omega_r, rated_power_gen, K_controller)
    # Calculate the derivative for the transition from time_points[i] to time_points[i+1]
    if current_omega_r == 0:
        d_omega_r_dt = 0
    else:
        d_omega_r_dt = (Pr_current - (Pg_current / eta_g)) / (J)
    new_omega_r = current_omega_r + d_omega_r_dt * time_step
    # Update current_omega_r for the next iteration
    current_omega_r = new_omega_r
    # Calculate other values for the *next* time point (time_points[i+1])
    next_time_step = time_points[i+1]
    U_next = wind_speed_interp(next_time_step) # Wind speed at the next time pointp
    lambda_tsr_next = (new_omega_r * R) / U_next
    Cp_next = Cp_function(lambda_tsr_next, np.rad2deg(current_pitch_angle))
    Pr_next = 0.5 * rho * Ar * Cp_next * U_next**3
    Pg_next = calculate_generator_power(U_next,new_omega_r, rated_power_gen, K_controller)
    power_next = Pg_next * eta_g
    # Append these newly calculated values for time_points[i+1]
    rotor_speeds.append(new_omega_r)
    power_turb.append(power_next)
    rews_values.append(U_next)
    pitch_angles.append(current_pitch_angle) # Append the updated pitch angle
# --- 7. Plotting Results ---
plt.figure(figsize=(12, 12)) # Increased figure height

plt.subplot(5, 1, 1) # Changed to 5 subplots
plt.plot(time_points, rews_values, label='Rotor-Effective Wind Speed (U)')
plt.ylabel('Wind Speed [m/s]')
plt.title('Wind Turbine Dynamics Simulation with Pitch Control')
plt.grid(True)
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(time_points, [r * 30 / np.pi for r in rotor_speeds], label='Rotor Speed [RPM]', color='orange')
plt.ylabel('Rotor Speed [RPM]')
plt.grid(True)
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(time_points, np.array(power_turb) / 1e6, label='Turbine Output (P_out) [MW]', color='blue')
plt.ylabel('Power [MW]')
plt.grid(True)
plt.legend()

plt.subplot(5, 1, 4) # New subplot for pitch angle
plt.plot(time_points, np.rad2deg(np.array(pitch_angles)), label='Blade Pitch Angle [deg]', color='purple')
plt.xlabel('Time [s]')
plt.ylabel('Pitch Angle [deg]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Simulation completed for {2*R}m diameter turbine with J={J} kg m^2.")
print(f"Initial Wind Speed: {initial_U:.2f} m/s")
print(f"Initial Rotor Speed: {rotor_speeds[0]:.2f} rad/s ({rotor_speeds[0] * 30 / np.pi:.2f} RPM)")
print(f"Rated Rotor Speed (target for pitch control): {rated_rotor_speed:.2f} rad/s ({rated_rotor_speed * 30 / np.pi:.2f} RPM)")
print(f"Controller K value: {K_controller:.2f}")
print(f"Cp_opt used for K_controller from table (at 0 deg pitch): {Cp_opt_for_K_calculation:.4f}")
print(f"Pitch Controller Gains: Kp={Kp_pitch}, Ki={Ki_pitch}")