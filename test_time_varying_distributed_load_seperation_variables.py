"""This is a script to test a time-varying distributed load with beam dynamics using analytical solution.

Uses separation of variables and modal analysis for the Euler-Bernoulli beam equation.

# FIXME: The physics model is entirely incorrect.

# FROM: Github Copilot, Claude Sonnet 4

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
from scipy.optimize import fsolve

# ------ PARAMETERS ------

# Beam length
L = 10.0

# Simulation time
T = 10.0

# Number of points and time steps
npoints, ntime_steps = 101, 500

# Beam properties (realistic steel cantilever)
rho = 7850      # Density (kg/m^3) - steel
b = 0.1         # Width (m)
h = 0.05        # Height (m)
A = b * h       # Cross-sectional area (m^2)
I = b * h**3 / 12  # Second moment of area (m^4)
E = 210e9       # Young's modulus (Pa)

# Number of modes to include
n_modes = 10

# ------ ANALYTICAL SOLUTION USING SEPARATION OF VARIABLES ------

def solve_cantilever_modes():
    """Solve for cantilever beam mode shapes and natural frequencies."""
    
    # Characteristic equation: cosh(λL)cos(λL) + 1 = 0
    def char_equation(lam):
        return np.cosh(lam * L) * np.cos(lam * L) + 1
    
    # Find roots (eigenvalues)
    lambda_n = []
    # Initial guesses for first few roots
    initial_guesses = [1.875/L, 4.694/L, 7.855/L, 10.996/L, 14.137/L]
    
    for i in range(n_modes):
        if i < len(initial_guesses):
            guess = initial_guesses[i]
        else:
            # For higher modes: λ_n ≈ (2n+1)π/(2L)
            guess = (2*i + 1) * np.pi / (2 * L)
        
        try:
            root = fsolve(char_equation, guess)[0]
            lambda_n.append(root)
        except:
            # Fallback approximation
            lambda_n.append((2*i + 1) * np.pi / (2 * L))
    
    return np.array(lambda_n)

def mode_shape(x, lam_n, L):
    """Calculate the nth mode shape for cantilever beam."""
    lam = lam_n
    
    # Constants for cantilever beam
    alpha_n = (np.sinh(lam * L) + np.sin(lam * L)) / (np.cosh(lam * L) + np.cos(lam * L))
    
    # Mode shape
    phi = (np.cosh(lam * x) - np.cos(lam * x) - 
           alpha_n * (np.sinh(lam * x) - np.sin(lam * x)))
    
    return phi

def solve_beam_analytical(q_func, x, t):
    """Solve beam dynamics using modal analysis."""
    
    print("Calculating mode shapes and frequencies...")
    
    # Get eigenvalues (λ_n)
    lambda_n = solve_cantilever_modes()
    
    # Calculate natural frequencies
    omega_n = lambda_n**2 * np.sqrt(E * I / (rho * A))
    
    # Fix the format error
    freq_hz = omega_n[:3] / (2 * np.pi)
    print(f"First 3 natural frequencies: {[f'{f:.2f}' for f in freq_hz]} Hz")
    
    # Initialize solution arrays
    w_total = np.zeros((len(t), len(x)))
    
    # Modal analysis - superposition of modes
    for n in range(n_modes):
        print(f"Processing mode {n+1}/{n_modes}...")
        
        # Calculate mode shape
        phi_n = mode_shape(x, lambda_n[n], L)
        
        # Normalize mode shape (mass normalization) - use numpy.trapz
        mass_norm = np.trapz(rho * A * phi_n**2, x)
        phi_n = phi_n / np.sqrt(mass_norm)
        
        # Calculate modal load (generalized force)
        def modal_load(t_val):
            q_t = q_func(x, t_val)
            return np.trapz(q_t * phi_n, x)
        
        # Solve modal equation: q_n''(t) + ω_n² q_n(t) = F_n(t) / M_n
        # For harmonic loading, we can solve analytically
        q_n = np.zeros(len(t))
        
        # Assume zero initial conditions
        for i, t_val in enumerate(t):
            # For sinusoidal load, use analytical solution
            # This is simplified - for general loading, use numerical integration
            F_n = modal_load(t_val)
            
            if i == 0:
                q_n[i] = 0
            else:
                # Simple integration (Newmark-β or similar would be better)
                dt = t[1] - t[0]
                # Simplified: assume constant load over timestep
                if omega_n[n] > 0:
                    # Undamped oscillator response
                    q_n[i] = (F_n / omega_n[n]**2) * (1 - np.cos(omega_n[n] * t_val))
        
        # Add contribution of this mode to total response
        for i in range(len(t)):
            w_total[i, :] += q_n[i] * phi_n
    
    return w_total, omega_n, lambda_n

def uniform_distributed_load(x, t):
    """Define a time-varying uniform distributed load."""
    q0 = 5000.0  # N/m
    
    # Gradual ramp + oscillation
    ramp = np.tanh(2 * t)
    load_magnitude = q0 * (0.5 + 0.3 * np.sin(2 * np.pi * t / T)) * ramp
    
    return np.full_like(x, load_magnitude)

# ------ SCRIPT ------

# Create grids
x = np.linspace(0, L, npoints)
t = np.linspace(0, T, ntime_steps)

print("Beam parameters:")
print(f"Length: {L} m")
print(f"Cross-section: {b} × {h} m")
print(f"E = {E/1e9:.0f} GPa, ρ = {rho} kg/m³")
print(f"EI = {E*I:.2e} N⋅m²")

# Solve using analytical method
print("\nSolving using analytical modal analysis...")
deflection, natural_freq, eigenvalues = solve_beam_analytical(uniform_distributed_load, x, t)

# Calculate velocity and stress
velocity = np.gradient(deflection, t[1] - t[0], axis=0)

# Calculate stress (bending stress at outer fiber)
stress = np.zeros_like(deflection)
dx = x[1] - x[0]
for i in range(len(t)):
    for j in range(1, len(x)-1):
        curvature = (deflection[i, j+1] - 2*deflection[i, j] + deflection[i, j-1]) / dx**2
        stress[i, j] = E * curvature * h / 2  # Outer fiber stress

print(f"\nResults:")
print(f"Max deflection: {np.max(np.abs(deflection)):.6f} m")
print(f"Max velocity: {np.max(np.abs(velocity)):.6f} m/s")
print(f"Max stress: {np.max(np.abs(stress))/1e6:.2f} MPa")
print(f"Deflection at tip: {deflection[-1, -1]:.6f} m")

# ------ PLOTTING ------

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Initialize lines
line1, = ax1.plot(x, deflection[0, :], 'b-', linewidth=2, label='Deflection')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

line2, = ax2.plot(x, velocity[0, :], 'm-', linewidth=2, label='Velocity')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

line3, = ax3.plot(x, stress[0, :]/1e6, 'r-', linewidth=2, label='Stress')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

line4, = ax4.plot(x, uniform_distributed_load(x, t[0])/1000, 'g-', linewidth=2, label='Load')

# Set up axes
ax1.set_ylabel('Deflection (m)')
ax1.set_title('Cantilever Beam: Analytical Solution')
ax1.grid(True)
ax1.legend()

ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Deflection Velocity')
ax2.set_xlabel('Position (m)')
ax2.grid(True)
ax2.legend()

ax3.set_ylabel('Stress (MPa)')
ax3.set_title('Bending Stress')
ax3.set_xlabel('Position (m)')
ax3.grid(True)
ax3.legend()

ax4.set_ylabel('Load (kN/m)')
ax4.set_title('Distributed Load')
ax4.set_xlabel('Position (m)')
ax4.grid(True)
ax4.legend()

# Set proper y-limits
max_defl = np.max(np.abs(deflection))
max_vel = np.max(np.abs(velocity))
max_stress = np.max(np.abs(stress)) / 1e6

ax1.set_ylim(-max_defl*0.1, max_defl*1.1)
ax2.set_ylim(-max_vel*1.1, max_vel*1.1)
ax3.set_ylim(-max_stress*1.1, max_stress*1.1)
ax4.set_ylim(0, 6)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, L)

def update(frame):
    if frame < len(t):
        line1.set_ydata(deflection[frame, :])
        line2.set_ydata(velocity[frame, :])
        line3.set_ydata(stress[frame, :] / 1e6)
        line4.set_ydata(uniform_distributed_load(x, t[frame]) / 1000)
        
        fig.suptitle(f'Cantilever Beam Analytical Solution - Time: {t[frame]:.2f} s\n'
                    f'Max deflection: {np.max(np.abs(deflection[frame, :])):.4f} m', 
                    fontsize=12)
    
    return line1, line2, line3, line4

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False, repeat=True)
plt.tight_layout()

# Show mode shapes
fig2, ax_modes = plt.subplots(figsize=(10, 6))
lambda_n = eigenvalues
for i in range(min(4, n_modes)):
    phi = mode_shape(x, lambda_n[i], L)
    phi = phi / np.max(np.abs(phi))  # Normalize for plotting
    ax_modes.plot(x, phi, label=f'Mode {i+1}: f = {natural_freq[i]/(2*np.pi):.2f} Hz')

ax_modes.set_xlabel('Position (m)')
ax_modes.set_ylabel('Normalized Mode Shape')
ax_modes.set_title('Cantilever Beam Mode Shapes')
ax_modes.grid(True)
ax_modes.legend()

plt.show()

