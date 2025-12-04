import numpy as np
import matplotlib.pyplot as plt

R = 126 / 2


chord = lambda f: 1 + 4 * np.exp(-0.001 * (f - 15) ** 2)

r = np.linspace(0, R, 1000)
plt.plot(r, chord(r))
plt.xlabel('Radius (m)')
plt.ylabel('Chord length (m)')
plt.xlim(0, R)
plt.ylim(0, None)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Blade Chord Distribution')
plt.grid()
plt.show()
