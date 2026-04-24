# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.

# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

# ====== BART ======
from typing import Literal
# ====== BART ======

import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import off.turbine as tur
import off.utils as util
import logging
from scipy.interpolate import interpn

# ====== BART ======
import sys, os
# Add the parent directory to the system path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert
# ====== BART ======

lg = logging.getLogger(__name__)


class Controller(ABC):
    settings: dict

    def __init__(self, settings: dict):
        """
        Class to decide the turbine states for the next time step based on the current states of turbines,
        ambient states and/or external input.

        Parameters
        ----------
        settings
        """

        self.settings = settings

    @abstractmethod
    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass

    @abstractmethod
    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    @abstractmethod
    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class IdealGreedyBaselineController(Controller):
    """
    Follows the main wind direction, disregarding yaw travel costs or other turbines
    """
    def __init__(self, settings: dict):
        super(IdealGreedyBaselineController, self).__init__(settings)
        lg.info('Ideal greedy baseline controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        turbine.set_yaw(turbine.ambient_states.get_wind_dir_ind(0), 0.0)

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float) -> pd.DataFrame:
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'time']
        )

        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class RealisticGreedyBaselineController(Controller):
    """
    Follows the main wind direction, only corrects if offset to the averaged wind direction is larger then a given
    offset.
    """

    def __init__(self, settings: dict):
        super(RealisticGreedyBaselineController, self).__init__(settings)
        self.moving = np.full((settings['number of turbines'], 1), False)
        self.moved = np.full((settings['number of turbines'], 1), False)
        self.time_steps_since_update = 0
        self.run_controller = True
        lg.info('Realistic greedy baseline controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)
        yaw = turbine.calc_yaw(wind_dir)
        self.moved[i_t] = self.moving[i_t]

        # check if difference is larger than threshold or if the turbine is moving already
        if np.abs(yaw) >= self.settings['misalignment_thresh'] and self.run_controller:
            # Turbine can move
            self.moving[i_t] = True

        if self.moving[i_t]:
            yaw_delta_t = np.sign(yaw) * self.settings['time step'] * turbine.yaw_rate_lim
            i_correction = np.argmin(
                np.abs([yaw, yaw_delta_t]))

            if i_correction == 0:
                # yaw angle 0.0 deg can be achieved
                turbine.set_yaw(wind_dir, 0.0)
                self.moved[i_t] = True  # in case the yaw misalignment was so small that it could be reached immediately
                self.moving[i_t] = False
            elif i_correction == 1:
                # yaw angle 0.0 deg can not be achieved, turbine keeps correcting
                turbine.set_yaw(wind_dir, yaw - yaw_delta_t)
        else:
            turbine.set_yaw(wind_dir, yaw)

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                self.moving[i_t],
                self.moved[i_t],
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'moving', 'moved', 'time']
        )

        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        # Reset
        if self.run_controller:
            self.run_controller = False

        # Check if condition is fulfilled
        self.time_steps_since_update = self.time_steps_since_update + 1
        if self.time_steps_since_update >= self.settings["apply_frequency"]:
            self.run_controller = True
            self.time_steps_since_update = 0


class YawSteeringLUTController(Controller):
    """
    Chooses yaw angles based on the current main wind direction and a lut for it. Only corrects if offset to the
    averaged wind direction is larger then a given offset.
    """

    def __init__(self, settings: dict):
        super(YawSteeringLUTController, self).__init__(settings)
        self.moving = np.full((settings['number of turbines'], 1), False)
        self.moved = np.full((settings['number of turbines'], 1), False)
        self.time_steps_since_update = 0
        self.run_controller = True
        self.lut = pd.read_csv(settings['path_to_angles_and_directions_csv'])
        # Mirror 0 deg index to 360 for binning
        self.lut = pd.concat([self.lut,
                              self.lut.loc[[0]].assign(**{'wind_direction': 360})], ignore_index=True)
        self.orientation = settings['orientation']
        lg.info('Yaw steering LUT controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        # Get wind direction
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)

        # Determine LUT bin
        i_bin = abs(self.lut['wind_direction'] - wind_dir).idxmin()

        # Check how long it has been in bin
        # TODO Time check skipped for now

        # Apply orientation
        lut_row = self.lut.loc[i_bin]
        # Convert the array of optimized values, read as string in the dataframe, into numbers TODO read as array
        if not self.orientation:
            opt_yaw = np.array([float(value) for value in self.lut.loc[0]['yaw_angles_opt'].strip('[]').split()])
            turbine.set_orientation_yaw(
                util.ot_get_orientation(
                    lut_row['wind_direction'],
                    opt_yaw[i_t]),
                wind_dir)

        # TODO apply LUT for orientation

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)
        i_bin = abs(self.lut['wind_direction'] - wind_dir).idxmin()

        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                wind_dir,
                i_bin,
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'wind_dir', 'i_bin', 'time']
        )
        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class YawSteeringPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines
    """

    def __init__(self, settings: dict):
        super(YawSteeringPrescribedMotionController, self).__init__(settings)

        # Read table
        if settings['input_method'] == "csv":
            t_and_ori = np.genfromtxt(settings['path_to_orientation_csv'], delimiter=',')
            self.lut = t_and_ori[:,1:]
            self.t   = t_and_ori[:,0]
        elif settings['input_method'] == "yaml":
            self.lut = np.array(settings['orientation_deg'])
            self.t   = settings['orientation_t']
        else:
            raise Warning("Orientation-input %s is undefined!" % settings['path_to_angles_and_directions_csv'])
        
        # TODO Check if complete
        # - check for number of turbines and number of columns in self.lut

        lg.info('Prescribed yaw motion controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        ori = np.interp(time_step, self.t, self.lut[:, i_t])
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)

        turbine.set_orientation_yaw(ori, wind_dir)

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """

        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)

        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                wind_dir,
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'wind_dir', 'time']
        )
        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class YawSteeringFilteredPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines, but filters and applies the hysteresis control
    """

    def __init__(self, settings: dict):
        super(YawSteeringFilteredPrescribedMotionController, self).__init__(settings)
        lg.info('Prescribed yaw motion controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class DeadbandYawSteeringLuTController(Controller):
    """
    Chooses yaw angles based on the current main wind direction and a lut for it. 
    Only corrects if offset to the averaged wind direction is larger then a given offset of if there has been a mismatch in wind direction for long enough.
    """

    def __init__(self, settings: dict, dt: float, nT: int):
        super(DeadbandYawSteeringLuTController, self).__init__(settings)

        # Get LuT
        if settings['path_to_angles_and_directions_pckl']:
            # Read pckl file
            df_opt = pd.read_pickle(settings['path_to_angles_and_directions_pckl'])

            # available values
            self.wind_vel = np.sort(df_opt['wind_speed'].value_counts().index)
            self.wind_dir = np.sort(df_opt['wind_direction'].value_counts().index)
            self.wind_ti  = np.sort(df_opt['turbulence_intensity'].value_counts().index)
            
            n_wind_dir = len(self.wind_dir)
            n_wind_vel = len(self.wind_vel)
            n_wind_ti  = len(self.wind_ti)
            n_wt       = len(df_opt['yaw_angles_opt'][0]) 

            self.lut = np.concatenate(df_opt['yaw_angles_opt'].to_numpy()).reshape((n_wind_dir, n_wind_vel, n_wind_ti, n_wt ))

        else:
            # Read table
            phi_and_ori = np.genfromtxt(settings['path_to_angles_and_directions_csv'], delimiter=',')
            self.lut    = phi_and_ori[:,1:]
            self.phi    = phi_and_ori[:,0]
            TypeError("Controller input: As of now, the csv file in not expected to provide the wind speed and turbulence intensity values. Please provide a pckl file instead.")

        self.wind_dir_thresh    = settings['wind_dir_thresh']
        self.k_i                = settings['k_i'] * dt
        self.integrated_error   = np.zeros(nT)
        self.set_wind_dir       = np.zeros(nT)
        self.run_controller     = True
        self.first_run          = np.full((nT, 1), False)
        self.dt                 = dt
        self.trigger_dir        = np.full((nT, 1), False)
        self.trigger_int        = np.full((nT, 1), False)
        self.orientation_lut    = np.zeros(nT)
        self.baseline           = settings['baseline']

        lg.info('Dead-band yaw steering LUT controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables & integrated error)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        # Get wind direction and orientation
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)
        wind_vel = turbine.ambient_states.get_turbine_wind_speed_abs()
        wind_ti = 0.06 #turbine.ambient_states.get_turbulence_intensity(0)
        orientation = turbine.get_yaw_orientation()

        # Init wind direction setting
        if self.first_run[i_t]:
            self.set_wind_dir[i_t] = wind_dir
            self.first_run[i_t] = False

        # Update integrated error
        self.integrated_error[i_t] += (wind_dir - self.set_wind_dir[i_t]) * self.k_i


        # check if difference is larger than threshold or if the turbine is moving already
        self.trigger_dir[i_t] = np.abs(self.set_wind_dir[i_t] - wind_dir) > self.wind_dir_thresh
        self.trigger_int[i_t] = self.integrated_error[i_t] > self.wind_dir_thresh

        if (self.trigger_dir[i_t] or self.trigger_int[i_t]) and self.run_controller:
            lg.info('Controller update triggered!')
            # Update set wind direction
            self.set_wind_dir[i_t] = wind_dir
            # Reset integrated error
            self.integrated_error[i_t] = 0

        # TODO add other thresholds for velocity and turbulence intensity
        
        if self.baseline:
            yaw_lut = 0.0
        else:
             # Determine yaw angle based on LUT
            yaw_lut = interpn((self.wind_dir, self.wind_vel, self.wind_ti), 
                        self.lut, 
                        np.array([self.set_wind_dir[i_t], wind_vel, wind_ti]).T, bounds_error=False, method='linear',fill_value=None).flatten()[i_t]

        self.orientation_lut[i_t] = util.ot_get_orientation(self.set_wind_dir[i_t], yaw_lut)

        # Move towards yaw angle and complete if possible
        delta_ori = self.orientation_lut[i_t] - orientation
        if abs(delta_ori) <= turbine.yaw_rate_lim * self.dt:
            # Achievable in one step
            turbine.set_orientation_yaw(self.orientation_lut[i_t], wind_dir)
        else:
            # Not achievable in one step
            turbine.set_orientation_yaw(orientation + np.sign(delta_ori) * turbine.yaw_rate_lim * self.dt, wind_dir)


    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.get_yaw_orientation(),
                self.orientation_lut[i_t],
                self.integrated_error[i_t],
                self.trigger_dir[i_t][0],
                self.trigger_int[i_t][0],
                self.set_wind_dir[i_t],
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'reference orientation','integrated error', 'direction trigger', 'integration trigger', 'set wind dir', 'time']
        )

        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass

# ====== BART ======

from typing import Callable

from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

# TODO: Also implement the gain-scheduled pitch controller from "Gain-Scheduled H∞ Control for WECS via LMI Techniques and Parametrically Dependent Feedback Part II: Controller Design and Implementation"


class PowerController(ABC):

    @abstractmethod
    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
        """Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class DownregulationControllerLio(PowerController):
    """Controller to set the power setpoint for downregulation based on the work of Lio et al."""

    def __init__(self, rotor_setpoint_strategy: Literal['constant_omega', 'max_omega', 'constant_tsr', 'min_Ct'], setpoints: dict):
        self.rotor_setpoint_strategy: str = rotor_setpoint_strategy
        self.setpoints: dict[str, list[float]] = setpoints
        self.power_setpoint: list[float] = [np.nan for _ in range(len(setpoints['power_factor'][0]))]
        self.rotor_setpoint: list[float] = [np.nan for _ in range(len(setpoints['power_factor'][0]))]
        self.wind_speed_derated: list[float] = [np.nan for _ in range(len(setpoints['power_factor'][0]))]
        self.T_g: list[float] = [np.nan for _ in range(len(setpoints['power_factor'][0]))]
        self.pitch: list[float] = [np.nan for _ in range(len(setpoints['power_factor'][0]))]
        self.K_P_gen: float = 2E6
        self.K_I_gen: float = 1E4
        self.K_P_pitch: float = 1
        self.K_I_pitch: float = 0
        self.error: list[float] = [0]  # List to store the error values for the integral action of the greedy pitch controller
        self.anti_windup_window: int | None = 1

    def __call__(self, turbine: tur, i_t: int, time_step: float, wind_speed: float) -> tur:

        def get_power_setpoint(setpoints: dict[str, list[float]], time_step: float, i_t: int) -> float:
            """Receives a dictionary with two keys, 'power_factor' and 'power_t', which contain lists of the same length. The power factor list contains the power factor setpoint to be applied at the corresponding time in the power_t list. This function returns the power factor setpoint corresponding to the given time step and turbine index, using linear interpolation with the given time"""
            power_factor = np.interp(time_step, setpoints['power_t'], np.array(setpoints['power_factor'])[:, i_t])
            power_setpoint = power_factor * turbine.rated_power
            return power_setpoint

        #: Calculate the power setpoint
        power_setpoint = get_power_setpoint(self.setpoints, time_step, i_t)
        #: Calculate the derated wind speed
        wind_speed_derated = ((power_setpoint) / (1/2 * turbine.AIR_DENSITY * np.pi * (turbine.rotor_radius ** 2) * turbine.opt_Cp * turbine.generator_efficiency)) ** (1/3)
        #: Calculate the rotor speed setpoint based on the power setpoint and the chosen strategy
        match self.rotor_setpoint_strategy:
            case 'max_omega':
                rotor_setpoint = convert(turbine.rated_rotor_speed, 'RPM', 'rad/s')
            case 'constant_omega':
                rotor_setpoint = (turbine.opt_tsr * wind_speed_derated) / turbine.rotor_radius
            case 'constant_tsr':
                rotor_setpoint = (turbine.opt_tsr * wind_speed) / turbine.rotor_radius
            case 'min_Ct':
                # FIXME: This is all very much spaghetti code
                try:
                    _ = self._init_lut
                except AttributeError as _:
                    Ct_min, P_fraction_vals, u_vals = np.load('dr_omega_vals.npy', allow_pickle=True), np.load('dr_P_fraction_vals.npy', allow_pickle=True), np.load('dr_u_vals.npy', allow_pickle=True)
                    self.lut = LookupTable(Ct_min, [P_fraction_vals, u_vals])
                    self._init_lut = True
                rotor_setpoint = convert(self.lut(power_setpoint / turbine.rated_power, wind_speed), 'RPM', 'rad/s')
            case _:
                raise ValueError(f"Rotor setpoint strategy {self.rotor_setpoint_strategy} is not recognized.")
        #: Compute the error
        error_t = turbine.omega - rotor_setpoint
        #: Compute the pitch angle based on the error in rotor speed
        if wind_speed < wind_speed_derated:
            pitch = 0
        else: 
            #: Compute the anti-windup integral action
            if self.anti_windup_window is None or len(self.error) < self.anti_windup_window:
                integral_term = np.sum([self.error[i] * self.dt for i in range(-len(self.error), -1)])
            else:
                integral_term = np.sum([self.error[i] * self.dt for i in range(-self.anti_windup_window, -1)])
            pitch = self.K_P_pitch * error_t + self.K_I_pitch * integral_term
        #: Calculate the generator torque
        if wind_speed < wind_speed_derated:
            #: Compute the anti-windup integral action
            if self.anti_windup_window is None or len(self.error) < self.anti_windup_window:
                integral_term = np.sum([self.error[i] * self.dt for i in range(-len(self.error), -1)])
            else:
                integral_term = np.sum([self.error[i] * self.dt for i in range(-self.anti_windup_window, -1)])
            T_g = self.K_P_gen * error_t + self.K_I_gen * integral_term
        else:
            T_g = power_setpoint / rotor_setpoint
        #: Append the error to the list of errors
        self.error.append(error_t)
        #: Set the control actions to the turbine
        turbine.T_g = T_g
        turbine.pitch = pitch
        #: Set the power setpoint
        turbine.power_setpoint = power_setpoint

        #: Save the latest setpoint
        self.power_setpoint[i_t] = power_setpoint
        self.rotor_setpoint[i_t] = rotor_setpoint
        self.wind_speed_derated[i_t] = wind_speed_derated
        self.T_g[i_t] = T_g
        self.pitch[i_t] = pitch


class LookupTable:
    """Stores a lookup table of any number of dimensions and provides a method to query it with interpolation"""

    def __init__(self, data: NDArray, axes: list[NDArray], interpolation_method: Literal['linear', 'nearest'] = 'linear', extrapolation_method: Literal['nearest', 'constant', 'none'] = 'nearest', extrapolation_fill_value: float | None = None):
        if data.ndim != len(axes):
            raise ValueError("The number of dimensions of the data must match the number of axes.")
        for idx, size in enumerate(data.shape):
            if size != axes[idx].size:
                raise ValueError(f"Size of data dimension {idx} does not match size of corresponding axis.")
        self.data: NDArray = data
        self.axes: list[NDArray] = axes
        match self.data.ndim:
            case 1:
                # FIXME: Does not check if the sample points are monotonically increasing
                self.func: Callable[[float], float] = lambda x: np.interp(x, self.axes[0], self.data)
            case 2:
                func = RegularGridInterpolator((self.axes[0], self.axes[1]), self.data, method=interpolation_method, bounds_error=False, fill_value=extrapolation_fill_value)
                self.func: Callable[[float, float], float] = lambda x_1, x_2: func(np.array([[x_1, x_2]]))[0]
            case 3:
                self.func: Callable[[float, float, float], float] = lambda x, y, z: interpn((self.axes[0], self.axes[1], self.axes[2]), self.data, np.array([x, y, z]).T, method=interpolation_method, bounds_error=False, fill_value=extrapolation_fill_value if extrapolation_method == 'constant' else None)
            case _:
                raise NotImplementedError("Lookup tables with more than 3 dimensions are not implemented yet.")
            
    def __call__(self, *args, **kwargs) -> float:
        return self.func(*args, **kwargs)
    
    def plot(self) -> None:
        """Plots the lookup table. Only implemented for 1D and 2D lookup tables."""
        import matplotlib.pyplot as plt
        match self.data.ndim:
            case 1:
                plt.plot(self.axes[0], self.data)
                plt.xlabel('x_1')
                plt.ylabel('f(x_1)')
                plt.title('1D Lookup Table')
                plt.grid()
                plt.show()
            case 2:
                X, Y = np.meshgrid(self.axes[0], self.axes[1], indexing='ij')
                Z = self.data
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='inferno')
                ax.set_xlabel('x_1')
                ax.set_ylabel('x_2')
                ax.set_zlabel('f(x_1, x_2)')
                ax.set_title('2D Lookup Table')
                plt.show()
            case _:
                raise NotImplementedError("Plotting is only implemented for 1D and 2D lookup tables.")
    
# ====== BART ======