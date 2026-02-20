"""Script to test integration with Numba"""

import time

import numpy as np
from numpy.typing import NDArray
from numba import float64
from numba.experimental import jitclass

# ------ PARAMETERS ------

# Set the number of ambient conditions
num_conditions = 500

# Set the ambient parameters
wind_direction = np.linspace(0, 360, num_conditions)
wind_speed = np.linspace(5, 15, num_conditions)
turbulence_intensity = np.full(num_conditions, 0.16)

# Set the layout of the wind farm
layout = np.array([[0, 0], 
                   [630, 0], 
                   [1260, 0]])

# Set the location where to examine the wake profile
pos = np.tile(np.array([200, 0, 90]), (2000, 1))

# ------ CLASSES ------

@jitclass([('layout', float64[:, :]), ('rotor_diameter', float64), ('hub_height', float64), ('tilt', float64), ('Cp', float64[:, :]), ('Ct', float64[:, :]), ('k_star', float64)])
class WindFarmModelNumba:

    def __init__(self, layout: NDArray, rotor_diameter: float, hub_height: float, tilt: float, Cp: NDArray, Ct: NDArray) -> None:
        self.layout = layout.astype(np.float64)
        self.rotor_diameter = float(rotor_diameter)
        self.hub_height = float(hub_height)
        self.tilt = float(tilt)
        self.Cp = Cp.astype(np.float64)
        self.Ct = Ct.astype(np.float64)
        # FIXME: These are preset wake attributes; can be parameterized in the future.
        self.k_star = 0.05

    def get_power(self, wind_direction: NDArray, wind_speed: NDArray, turbulence_intensity: NDArray) -> NDArray:
        # FIXME: Placeholder. Note that commenting out `@jitclass` decorator does cause a significant increase in runtime, so the Numba integration is working (which is especially noticeable for for-loops).
        for wd in wind_direction:
            for ws in wind_speed:
                for ti in turbulence_intensity:
                    ...
        return None
    
    def get_vel(self, _wind_direction: NDArray, wind_speed: NDArray, _turbulence_intensity: NDArray, pos: NDArray) -> NDArray:
        pos = np.atleast_2d(pos)
        vel_def = np.zeros((wind_speed.size, pos.shape[0]))
        for idx_pos, loc in enumerate(pos):
            x, y, z = loc
            d_0, z_h = self.rotor_diameter, self.hub_height
            Ct = np.zeros_like(wind_speed)
            for idx in range(wind_speed.size):
                for idx_Ct in range(self.Ct.shape[0]):
                    if wind_speed[idx] <= self.Ct[idx_Ct, 0]:
                        Ct[idx] = self.Ct[idx_Ct, 1]
                        break
            eps = 0.1  # FIXME: Placeholder for regularization term
            # FROM: "Power-Load Balanced Wake Steering Control using a Dynamic Wind Farm Flow Model," H. Gielen (2026) | Eq. (2.1)
            vel_def[:, idx_pos] = (1 - np.sqrt(1 - (Ct / (8 * (self.k_star * (x / d_0) + eps) ** 2)))) * np.exp(-(1 / (2 * (self.k_star * (x / d_0) + eps) ** 2)) * (((z - z_h) / (d_0)) ** 2 + (y / d_0) ** 2))
        return vel_def


class WindFarmModel:
    def __init__(self, layout: NDArray, turbine_model: dict) -> None:
        rotor_diameter = float(turbine_model['rotor_diameter'])
        hub_height = float(turbine_model['hub_height'])
        tilt = float(turbine_model['tilt'])
        Cp = np.asarray(turbine_model['Cp'], dtype=np.float64)
        Ct = np.asarray(turbine_model['Ct'], dtype=np.float64)
        self._core = WindFarmModelNumba(layout, rotor_diameter, hub_height, tilt, Cp, Ct)

    def get_power(self, wind_direction: NDArray, wind_speed: NDArray, turbulence_intensity: NDArray) -> NDArray:
        return self._core.get_power(wind_direction, wind_speed, turbulence_intensity)
    
    def get_vel(self, wind_direction: NDArray, wind_speed: NDArray, turbulence_intensity: NDArray, pos: NDArray) -> NDArray:
        return self._core.get_vel(wind_direction, wind_speed, turbulence_intensity, pos)

# ------ FUNCTIONS ------

# ------ SCRIPT ------

# Initialize the wind turbine model (NREL 5MW)
wtm = {
    'rotor_diameter': 125.88,
    'hub_height': 90.0,
    'tilt': 5.0,
    'Cp': np.array(
        [[ 0. ,    0. ],
         [ 2.9,    0. ],
         [ 3. ,   40.5],
         [ 4. ,  177.7],
         [ 5. ,  403.9],
         [ 6. ,  737.6],
         [ 7. , 1187.2],
         [ 7.1, 1239.2],
         [ 7.2, 1292.5],
         [ 7.3, 1347.3],
         [ 7.4, 1403.3],
         [ 7.5, 1460.7],
         [ 7.6, 1519.6],
         [ 7.7, 1580.2],
         [ 7.8, 1642.1],
         [ 7.9, 1705.8],
         [ 8. , 1771.2],
         [ 9.0, 2518.6],
         [10. , 3448.4],
         [10.1, 3552.1],
         [10.2, 3658. ],
         [10.3, 3765.1],
         [10.4, 3873.9],
         [10.5, 3984.5],
         [10.6, 4096.6],
         [10.7, 4210.7],
         [10.8, 4326.2],
         [10.9, 4443.4],
         [11. , 4562.5],
         [11.1, 4683.4],
         [11.2, 4806.2],
         [11.3, 4929.9],
         [11.4, 5000. ],
         [11.5, 5000. ],
         [11.6, 5000. ],
         [11.7, 5000. ],
         [11.8, 5000. ],
         [11.9, 5000. ],
         [12. , 5000. ],
         [13. , 5000. ],
         [14. , 5000. ],
         [15. , 5000. ],
         [16. , 5000. ],
         [17. , 5000. ],
         [18. , 5000. ],
         [19. , 5000. ],
         [20. , 5000. ],
         [21. , 5000. ],
         [22. , 5000. ],
         [23. , 5000. ],
         [24. , 5000. ],
         [25. , 5000. ],
         [25.1,    0. ],
         [50. ,    0. ]]),
    'Ct': np.array(
        [[ 0. ,  0. ],
         [ 2.9,  0. ],
         [ 3. ,  1.1],
         [ 4. ,  1. ],
         [ 5. ,  0.9],
         [ 6. ,  0.9],
         [ 7. ,  0.8],
         [ 7.1,  0.8],
         [ 7.2,  0.8],
         [ 7.3,  0.8],
         [ 7.4,  0.8],
         [ 7.5,  0.8],
         [ 7.6,  0.8],
         [ 7.7,  0.8],
         [ 7.8,  0.8],
         [ 7.9,  0.8],
         [ 8. ,  0.8],
         [ 9. ,  0.8],
         [10. ,  0.8],
         [10.1,  0.8],
         [10.2,  0.8],
         [10.3,  0.8],
         [10.4,  0.8],
         [10.5,  0.8],
         [10.6,  0.8],
         [10.7,  0.8],
         [10.8,  0.8],
         [10.9,  0.8],
         [11. ,  0.8],
         [11.1,  0.8],
         [11.2,  0.7],
         [11.3,  0.7],
         [11.4,  0.7],
         [11.5,  0.7],
         [11.6,  0.6],
         [11.7,  0.6],
         [11.8,  0.6],
         [11.9,  0.6],
         [12. ,  0.5],
         [13. ,  0.4],
         [14. ,  0.3],
         [15. ,  0.2],
         [16. ,  0.2],
         [17. ,  0.2],
         [18. ,  0.1],
         [19. ,  0.1],
         [20. ,  0.1],
         [21. ,  0.1],
         [22. ,  0.1],
         [23. ,  0.1],
         [24. ,  0.1],
         [25. ,  0.1],
         [25.1,  0. ],
         [50. ,  0. ]])
}

# Initialize the wind farm model
wfm = WindFarmModel(layout, wtm)  # NOTE: For now, only supports homogeneous layouts

# Compute the power output
time_start_power = time.time()
power = wfm.get_power(wind_direction, wind_speed, turbulence_intensity)
time_end_power = time.time()

# Compute the velocity deficit at a point in the wake of the first turbine
time_start_vel = time.time()
vel_def = wfm.get_vel(wind_direction, wind_speed, turbulence_intensity, pos)
time_end_vel = time.time()

# ------ PRINTING ------

np.set_printoptions(precision=1, suppress=True, threshold=10)

print(f"\nPower output for each turbine:\n{power} (shape = {power.shape if power is not None else 'None'})")
print(f"Time taken to compute power output: {time_end_power - time_start_power:.4f} seconds")

print(f"\nVelocity deficit:\n{vel_def} (shape = {vel_def.shape})")
print(f"Time taken to compute velocity deficit: {time_end_vel - time_start_vel:.4f} seconds")