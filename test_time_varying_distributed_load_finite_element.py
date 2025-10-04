"""This is a script to test a time-varying distributed load with beam dynamics using finite element method.

Uses finite element analysis for the Euler-Bernoulli beam equation.

# FROM: Github Copilot, Claude Sonnet 4

# FIXME: Literally takes forever to run, probably also incorrect.

# FROM: https://teachbooks.tudelft.nl/computational-modelling/dynamics/Exercises/str_elem_dyn_workshops/Workshop_FEM_dyn_beam.html

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ------ PARAMETERS ------

# Beam length
L = 10.0

# Simulation time
T = 10.0

# Number of elements and time steps
n_elements = 50
ntime_steps = 500

# Beam properties (realistic steel cantilever)
rho = 7850      # Density (kg/m^3) - steel
b = 0.1         # Width (m)
h = 0.05        # Height (m)
A = b * h       # Cross-sectional area (m^2)
I = b * h**3 / 12  # Second moment of area (m^4)
E = 210e9       # Young's modulus (Pa)

# ------ FINITE ELEMENT IMPLEMENTATION ------

def hermite_shape_functions(xi):
    """Hermite cubic shape functions for beam element."""
    # xi is local coordinate from -1 to 1
    N1 = 0.25 * (1 - xi)**2 * (2 + xi)
    N2 = 0.125 * (1 - xi)**2 * (1 + xi)
    N3 = 0.25 * (1 + xi)**2 * (2 - xi)
    N4 = 0.125 * (1 + xi)**2 * (xi - 1)
    
    return np.array([N1, N2, N3, N4])

def hermite_derivatives(xi):
    """Derivatives of Hermite shape functions."""
    dN1 = -1.5 * (1 - xi) * (1 + xi)
    dN2 = 0.25 * (1 - xi) * (3*xi + 1)
    dN3 = 1.5 * (1 - xi) * (1 + xi)
    dN4 = 0.25 * (1 + xi) * (3*xi - 1)
    
    return np.array([dN1, dN2, dN3, dN4])

def hermite_second_derivatives(xi):
    """Second derivatives of Hermite shape functions."""
    d2N1 = -3 * xi
    d2N2 = 1.5 * xi - 0.5
    d2N3 = 3 * xi
    d2N4 = 1.5 * xi + 0.5
    
    return np.array([d2N1, d2N2, d2N3, d2N4])

def element_matrices(Le):
    """Calculate element mass and stiffness matrices."""
    # Gauss quadrature points and weights
    gauss_points = np.array([-0.7745966692, 0.0, 0.7745966692])
    weights = np.array([0.5555555556, 0.8888888889, 0.5555555556])
    
    Me = np.zeros((4, 4))  # Element mass matrix
    Ke = np.zeros((4, 4))  # Element stiffness matrix
    
    for i, xi in enumerate(gauss_points):
        N = hermite_shape_functions(xi)
        d2N_dxi2 = hermite_second_derivatives(xi)
        
        # Jacobian for coordinate transformation
        J = Le / 2
        
        # Mass matrix contribution
        Me += rho * A * np.outer(N, N) * weights[i] * J
        
        # Stiffness matrix contribution (curvature-based)
        # d²N/dx² = (2/Le)² * d²N/dxi²
        d2N_dx2 = (2/Le)**2 * d2N_dxi2
        Ke += E * I * np.outer(d2N_dx2, d2N_dx2) * weights[i] * J
    
    return Me, Ke

def assemble_global_matrices(n_elements):
    """Assemble global mass and stiffness matrices."""
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes  # 2 DOFs per node (deflection + rotation)
    
    M_global = np.zeros((n_dofs, n_dofs))
    K_global = np.zeros((n_dofs, n_dofs))
    
    Le = L / n_elements  # Element length
    
    for e in range(n_elements):
        Me, Ke = element_matrices(Le)
        
        # Global DOF indices for element e
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        
        # Assemble into global matrices
        for i in range(4):
            for j in range(4):
                M_global[dofs[i], dofs[j]] += Me[i, j]
                K_global[dofs[i], dofs[j]] += Ke[i, j]
    
    return M_global, K_global

def apply_boundary_conditions(M, K, F):
    """Apply cantilever boundary conditions (fixed at x=0)."""
    # Remove DOFs 0 and 1 (deflection and rotation at x=0)
    free_dofs = np.arange(2, M.shape[0])
    
    M_reduced = M[np.ix_(free_dofs, free_dofs)]
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    
    return M_reduced, K_reduced, F_reduced, free_dofs

def calculate_load_vector(q_func, t_val, n_elements):
    """Calculate global load vector from distributed load."""
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes
    F_global = np.zeros(n_dofs)
    
    Le = L / n_elements
    
    # Gauss quadrature
    gauss_points = np.array([-0.7745966692, 0.0, 0.7745966692])
    weights = np.array([0.5555555556, 0.8888888889, 0.5555555556])
    
    for e in range(n_elements):
        x_elem = np.linspace(e * Le, (e + 1) * Le, 2)
        Fe = np.zeros(4)
        
        for i, xi in enumerate(gauss_points):
            N = hermite_shape_functions(xi)
            x_global = x_elem[0] + 0.5 * (1 + xi) * Le
            q_val = q_func(x_global, t_val)
            
            # Only deflection DOFs get load (not rotational DOFs)
            load_contribution = q_val * N * weights[i] * Le / 2
            Fe[0] += load_contribution[0]  # Node 1 deflection
            Fe[2] += load_contribution[2]  # Node 2 deflection
        
        # Assemble into global load vector
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            F_global[dofs[i]] += Fe[i]
    
    return F_global

def solve_beam_deflection_stresses(q_func, x, t):
    """Solve beam dynamics using finite element method."""
    
    print("Setting up finite element system...")
    
    # Assemble global matrices
    M_global, K_global = assemble_global_matrices(n_elements)
    
    print(f"Global system size: {M_global.shape[0]} DOFs")
    
    # Apply boundary conditions once to get reduced system
    F_dummy = np.zeros(M_global.shape[0])
    M_red, K_red, F_dummy_red, free_dofs = apply_boundary_conditions(M_global, K_global, F_dummy)
    
    n_free_dofs = len(free_dofs)
    
    # Time integration setup
    def system_dynamics(t_val, y):
        """System of ODEs for beam dynamics."""
        # y contains [displacements, velocities] for free DOFs only
        displacements = y[:n_free_dofs]
        velocities = y[n_free_dofs:]
        
        # Calculate load vector
        F_global = calculate_load_vector(q_func, t_val, n_elements)
        F_red = F_global[free_dofs]
        
        # Equation: M*a = F - K*u
        accelerations = spsolve(csr_matrix(M_red), F_red - K_red @ displacements)
        
        # Return derivatives [velocities, accelerations]
        dydt = np.concatenate([velocities, accelerations])
        
        return dydt
    
    # Initial conditions (beam at rest) - only for free DOFs
    y0 = np.zeros(2 * n_free_dofs)
    
    # Solve ODE system
    print("Solving time integration...")
    sol = solve_ivp(system_dynamics, [t[0], t[-1]], y0, t_eval=t, method='RK45', rtol=1e-6)
    
    if not sol.success:
        print(f"Warning: Integration failed - {sol.message}")
    
    # Extract results
    n_nodes = n_elements + 1
    x_nodes = np.linspace(0, L, n_nodes)
    
    deflections = np.zeros((len(t), len(x)))
    velocities = np.zeros((len(t), len(x)))
    
    for i, t_val in enumerate(t):
        # Reconstruct full displacement vector
        u_full = np.zeros(M_global.shape[0])
        v_full = np.zeros(M_global.shape[0])
        
        # Insert free DOF values
        u_full[free_dofs] = sol.y[:n_free_dofs, i]
        v_full[free_dofs] = sol.y[n_free_dofs:, i]
        
        # Extract deflections at nodes (every other DOF)
        u_nodes = u_full[::2]  # Deflection DOFs
        v_nodes = v_full[::2]  # Velocity DOFs
        
        # Interpolate to requested x points
        deflections[i, :] = np.interp(x, x_nodes, u_nodes)
        velocities[i, :] = np.interp(x, x_nodes, v_nodes)
    
    # Calculate stresses
    stresses = np.zeros_like(deflections)
    dx = x[1] - x[0]
    for i in range(len(t)):
        for j in range(1, len(x)-1):
            curvature = (deflections[i, j+1] - 2*deflections[i, j] + deflections[i, j-1]) / dx**2
            stresses[i, j] = E * curvature * h / 2
    
    print("Finite element solution completed!")
    
    return deflections, velocities, stresses

def uniform_distributed_load(x, t):
    """Define a time-varying uniform distributed load."""
    q0 = 5000.0  # N/m
    
    # Gradual ramp + oscillation
    ramp = np.tanh(2 * t)
    load_magnitude = q0 * (0.5 + 0.3 * np.sin(2 * np.pi * t / T)) * ramp
    
    return load_magnitude  # Return scalar for single point

# ------ SCRIPT ------

# Create grids
x = np.linspace(0, L, 101)
t = np.linspace(0, T, ntime_steps)

print("Beam parameters:")
print(f"Length: {L} m")
print(f"Cross-section: {b} × {h} m")
print(f"E = {E/1e9:.0f} GPa, ρ = {rho} kg/m³")
print(f"EI = {E*I:.2e} N⋅m²")
print(f"Number of elements: {n_elements}")

# Solve using finite element method
print("\nSolving using finite element method...")
deflection, velocity, stress = solve_beam_deflection_stresses(uniform_distributed_load, x, t)

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

line4, = ax4.plot(x, [uniform_distributed_load(xi, t[0]) for xi in x], 'g-', linewidth=2, label='Load')

# Set up axes
ax1.set_ylabel('Deflection (m)')
ax1.set_title('Cantilever Beam: Finite Element Solution')
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

ax4.set_ylabel('Load (N/m)')
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
ax4.set_ylim(0, 6000)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, L)

def update(frame):
    if frame < len(t):
        line1.set_ydata(deflection[frame, :])
        line2.set_ydata(velocity[frame, :])
        line3.set_ydata(stress[frame, :] / 1e6)
        line4.set_ydata([uniform_distributed_load(xi, t[frame]) for xi in x])
        
        fig.suptitle(f'Cantilever Beam FE Solution - Time: {t[frame]:.2f} s\n'
                    f'Max deflection: {np.max(np.abs(deflection[frame, :])):.6f} m', 
                    fontsize=12)
    
    return line1, line2, line3, line4

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False, repeat=True)
plt.tight_layout()

plt.show()

