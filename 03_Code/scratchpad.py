import control as ct

omg_n = 11.11  # natural frequency of the pitch actuator (in rad/s)
eta = 0.6  # damping ratio of the pitch actuator

s = ct.tf('s')
G_pitch = omg_n ** 2 / (s ** 2 + 2 * eta * omg_n * s + omg_n ** 2)  # transfer function of the pitch actuator

sys = ct.tf2ss(G_pitch)  # convert to state-space representation
sys, T = ct.canonical_form(sys, 'observable')  # convert to controllable canonical form
A, B, C, D = sys.A, sys.B, sys.C, sys.D

print(A)
print(B)
print(C)
print(D)