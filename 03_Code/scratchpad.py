import matplotlib.pyplot as plt
import numpy as np

Cp = np.loadtxt('Cp.csv', delimiter=';')
Ct = np.loadtxt('Ct.csv', delimiter=';')

Cp = np.flipud(Cp)
Ct = np.flipud(Ct)

# plt.imshow(Cp, aspect='equal', cmap='inferno')
# plt.show()

# plt.imshow(Ct, aspect='equal', cmap='inferno')
# plt.show()

pitch_range = np.linspace(-10, 50, Cp.shape[1] + 1)
pitch_range = np.delete(pitch_range, np.isclose(pitch_range, -0.5))  # Remove the first element 

Pitch, TSR = np.meshgrid(pitch_range, np.linspace(15, 0.05, Cp.shape[0]), indexing='xy')

print(pitch_range)

Pitch = np.flipud(Pitch)
TSR = np.flipud(TSR)
Cp = np.flipud(Cp)
Ct = np.flipud(Ct)

Data = np.column_stack((np.arange(Cp.size), Pitch.flatten(order='F'), TSR.flatten(order='F'), Cp.flatten(order='F'), Ct.flatten(order='F')))

np.savetxt('Cp_Ct_NREL5MW_nrel.csv', Data, delimiter=';', header='idx;pitch;tsr;Cp;Ct', comments='', fmt=['%d', '%.2f', '%.2f', '%.6f', '%.6f'])

Data = np.loadtxt('Cp_Ct_NREL5MW_nrel.csv', delimiter=';', skiprows=1)
pitch, tsr, Cp_load, Ct_load = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]]

plt.imshow(Cp_load, aspect='equal', cmap='inferno')
plt.show()

plt.imshow(Ct_load, aspect='equal', cmap='inferno')
plt.show()

plt.imshow(pitch, aspect='equal', cmap='inferno')
plt.show()

plt.imshow(tsr, aspect='equal', cmap='inferno')
plt.show()