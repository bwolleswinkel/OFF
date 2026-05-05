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

import warnings
# ====== BART ======
from pathlib import Path
from typing import Callable, Literal, Optional
import scipy as sp
# ====== BART ======

import numpy as np
from abc import ABC, abstractmethod
from off.observation_points import ObservationPoints
from off.ambient import AmbientStates
from off.states import States
import off.utils as ot
import logging
lg = logging.getLogger(__name__)

# ====== BART ======
import sys, os
# Add the parent directory to the system path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert
# ====== BART ======


class TurbineStates(States, ABC):

    def __init__(self, number_of_time_steps: int, number_of_states: int, state_names: list):
        """
        Abstract base class for the turbine states, inherits from abstract States class. This class determines how many
        turbine states are stored and how they are used to calculate the Cp and Ct coefficient. The states class provides
        the get, set, init & iterate methods

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        number_of_states : int
            number of states per time step
        state_names : list
            name and unit of the states
        """
        super(TurbineStates, self).__init__(number_of_time_steps, number_of_states, state_names)
        lg.info('Turbine states chain created with %s time steps and %s states' %
                (number_of_time_steps, number_of_states))
        lg.info(state_names)

    @abstractmethod
    def get_current_cp(self) -> float:
        """
        get_current_cp returns the current power coefficient of the turbine

        Returns
        -------
        float:
            Power coefficient (-)
        """
        pass

    @abstractmethod
    def get_current_ct(self) -> float:
        """
        get_current_ct returns the current thrust coefficient of the turbine

        Returns
        -------
        float:
            Thrust coefficient (-)
        """
        pass

    @abstractmethod
    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location

        Returns
        -------
        float:
            yaw misalignment at the turbine location (deg)
        """
        pass

    @abstractmethod
    def get_current_ax_ind(self) -> float:
        """
        get_current_axInd returns the current axial induction factor of the turbine

        Returns
        -------
        float:
            Axial induction factor (-)
        """
        pass

    @abstractmethod
    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which Ct should be calculated
        Returns
        -------
        float:
            Thrust coefficient
        """
        pass

    @abstractmethod
    def get_ax_ind(self, index: int) -> float:
        """
        get_ax_ind(index) returns the axial induction factor at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which a should be calculated
        Returns
        -------
        float:
            Axial induction factor
        """
        pass

    @abstractmethod
    def set_ax_ind(self, ax_ind):
        """
        Stores the axial induction factor of the turbine in the states.

        Parameters
        ----------
        ax_ind: axial induction factor
        """
        pass

    @abstractmethod
    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index

        Parameters
        ----------
        index : int

        Returns
        -------
        float:
            yaw misalignment in deg
        """
        pass

    @abstractmethod
    def set_yaw(self, yaw_angle):
        """
        Stores the yaw angle of the turbine in the states.

        Parameters
        ----------
        yaw_angle: Difference between turbine orientation and wind direction
        """
        pass

    @abstractmethod
    def get_all_ct(self) -> np.ndarray:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.

        Returns
        -------
        np.ndarray:
            Thrust coefficient at all turbine states (-)
        """
        pass

    @abstractmethod
    def get_all_yaw(self) -> np.ndarray:
        """
        get_all_ct(index) returns the yaw misalignment for all turbine states.

        Returns
        -------
        np.ndarray:
            Yaw misalignment at all turbine states (deg)
        """
        pass

    @abstractmethod
    def get_all_ax_ind(self) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        pass

    @abstractmethod
    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates a TurbineStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_current_yaw()

        Parameters
        ----------
        index1 : int
            Index of the first state
        index2 : int
            Index of the second state
        w1 : float
            Weight for first index (has to be w1 = 1 - w2, and [0,1])
        w2 : float
            Weight for second index (has to be w2 = 1 - w1, and [0,1])

        Returns
        -------
        TurbineStates
            turbine state object with single entry
        """
        pass


class Turbine(ABC):
    # Attributes
    diameter = 1  # in Meter
    nacellePos = np.array([0, 0, 1])            # in Meter
    turbine_type = "base"
    orientation = np.array([0, 0])              # yaw, tilt in Degree

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates):
        """
        Turbine abstract base class

        Parameters
        ----------
        base_location : np.ndarray
            1 x 3 vector with x, y, z position of the turbine base (m)
        orientation : np.ndarray
            1 x 2 vector with yaw, tilt orientation of the turbine
        turbine_states : TurbineStates
            Turbine state object
        observation_points : ObservationPoints
            Observation Points object
        ambient_states : AmbientStates
            Ambient states object
        """
        self.base_location = base_location
        self.orientation = orientation
        self.turbine_states = turbine_states
        self.observation_points = observation_points
        self.ambient_states = ambient_states

    def calc_yaw(self, wind_direction: float) -> float:
        """
        Get the yaw misalignment of the turbine

        Parameters
        ----------
        wind_direction : number
            Wind direction (deg)

        Returns
        -------
        float:
            yaw misalignment (deg)
        """
        return ot.ot_get_yaw(wind_direction, self.orientation[0])
    
    def get_yaw_orientation(self) -> float:
        """
        Get the yaw orientation of the turbine

        Returns
        -------
        float:
            yaw orientation (deg)
        """
        return self.orientation[0]

    def get_tilt_orientation(self) -> float:
        """
        Get the tilt orientation of the turbine

        Returns
        -------
        float:
            tilt orientation (deg)
        """
        return self.orientation[1]

    def set_yaw(self, wind_direction: float, yaw: float):
        """
        Sets the orientation based on the given wind direction and yaw angle

        Parameters
        ----------
        wind_direction : float
            Wind direction in degrees
        yaw : float
            Turbine yaw misalignment angle in degrees

        """
        self.orientation[0] = ot.ot_get_orientation(wind_direction, yaw)
        self.turbine_states.set_yaw(yaw)
        lg.debug("Turbine yaw angle set to %s deg, resulting orientation %s deg" % (yaw, self.orientation[0]))

    def set_orientation_yaw(self, orientation_yaw: float, wind_direction=orientation):
        """
        Sets the orientation of the turbine in yaw direction (opposed to tilt), calculates the effective yaw angle and
        updates the turbine states
        Parameters
        ----------
        orientation_yaw: float
            Turbine orientation in deg. If the turbine orientation is equal to the wind direction, yaw = 0
        wind_direction: float
            Wind direction in deg
        """
        self.orientation[0] = orientation_yaw
        yaw = self.calc_yaw(wind_direction)
        self.turbine_states.set_yaw(yaw)
        lg.debug("Turbine yaw orientation set to %s deg, resulting yaw angle %s deg" % (orientation_yaw, yaw))

    def set_tilt(self, tilt: float):
        """
        Sets the tilt angle of the turbine
        Parameters
        ----------
        tilt: float
            Tilt angle in deg

        Returns
        -------

        """
        self.orientation[1] = tilt

    def calc_tilt(self):
        """
            Get the tilt of the turbine

            Returns
            -------
            float:
                tilt (deg)
            """
        return self.orientation[1]

    @abstractmethod
    def calc_power(self, wind_speed, air_den):
        """
        Calculate the power based on turbine, ambient and OP states

        Parameters
        ----------
        wind_speed : float
            Wind speed (m/s)
        air_den : float
            air density

        Returns
        -------
        float :
            Power generated (W)
        """
        # TODO this function should either rely on its own states or be more descriptive with what inputs are need
        pass

    def get_rotor_pos(self) -> float:
        """
        Calculates the rotor position based on the current yaw and tilt

        Returns
        -------
        np.ndarray:
            1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        """
        # TODO add tilt to offset calculation
        lg.info('Orientation [0] %s ' % self.orientation[0])

        yaw = ot.ot_deg2rad(self.orientation[0])
        offset = np.array([np.cos(yaw), np.sin(yaw), 1]) * \
            self.nacellePos

        return self.base_location + offset

    def set_rotor_pos(self, pos_rot: np.ndarray):
        """
        Sets the base location, based on a rotor location and the nacelle position
        Parameters
        ----------
        pos_rot
        """
        # TODO should pay attention to orientation of the turbine
        self.base_location = pos_rot - self.nacellePos


class HAWT_ADM(Turbine):
    # Attributes
    diameter = 178.4  # Meter
    nacellePos = np.array([0, 0, 119])  # in Meter
    turbine_type = "name"

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates, turbine_data: dict):
        """
        HAWT extends the base turbine class and specifies a generic horizontal axis wind turbine

        Parameters
        ----------
        base_location : np.ndarray
            1 x 3 vector with x, y, z position of the turbine base (m)
        orientation : np.ndarray
            1 x 2 vector with yaw, tilt orientation of the turbine
        turbine_states : TurbineStates
            Turbine state object
        observation_points : ObservationPoints
            Observation Points object
        ambient_states : AmbientStates
            Ambient states object
        """
        self.diameter = turbine_data["rotor_diameter"]
        self.nacellePos = np.array([0, 0, turbine_data["hub_height"]])
        self.power_calc_method = "cp-u lut"  # alternative to "axial induction", "cp-bpa-tsr"  TODO: Set later in input
        self.thrust_calc_method = "ct-u lut"  # alternative to "axial induction", "ct-bpa-tsr" TODO: Set later in input
        self.yaw_power_coeff = "pP" # alternative to "none" TODO: Set later in input
        self.yaw_thrust_coeff ="pT" # alternative to "none" TODO: Set later in input

        if "rotor_overhang" in turbine_data:
            self.nacellePos[0] = turbine_data["rotor_overhang"]

        if "Cp_curve" in turbine_data["performance"]:
            self.Cp_u_values = turbine_data["performance"]["Cp_curve"]["Cp_u_values"]
            self.Cp_u_wind_speeds = turbine_data["performance"]["Cp_curve"]["Cp_u_wind_speeds"]

        if "Ct_curve" in turbine_data:
            self.Ct_u_values = turbine_data["performance"]["Ct_curve"]["Ct_u_values"]
            self.Ct_u_wind_speeds = turbine_data["performance"]["Ct_curve"]["Ct_u_values"]

        if "pP" in turbine_data:
            self.Cp_pP = turbine_data["pP"]

        if "pT" in turbine_data:
            self.Cp_pT = turbine_data["pT"]

        if "yaw_rate_lim" in turbine_data:
            self.yaw_rate_lim = turbine_data["yaw_rate_lim"]

        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states)
        lg.info("HAWT turbine of type " + turbine_data["name"] + "created")
        lg.info('Turbine base location: %s' % base_location)
        lg.info('Power calculation method: %s' % self.power_calc_method)
        lg.info('Thrust calculation method: %s' % self.thrust_calc_method)
        lg.info('Power yaw coefficient: %s' % self.yaw_power_coeff)
        lg.info('Thrust yaw coefficient: %s' % self.yaw_thrust_coeff)

    def calc_power(self, wind_speed, air_den=1.225):
        """
        Calculate the power based on turbine, ambient and OP states

        Parameters
        ----------
        wind_speed : float
            Wind speed (m/s)
        air_den : float
            air density

        Returns
        -------
        float :
            Power generated (W)
        """
        if self.yaw_power_coeff == "pP":
            yaw = np.deg2rad(self.turbine_states.get_current_yaw())
            yaw_coef = np.cos(yaw) ** self.Cp_pP
        elif self.yaw_power_coeff == "none":
            yaw_coef = 1.0
        else:
            raise Exception("Only cos(yaw) ** pP yaw coefficient supported (or none)")

        if self.power_calc_method == "axial induction":
            axi = self.turbine_states.get_current_ax_ind()
            cp = 4 * axi * (1 - axi) ** 2
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-u lut":
            cp = np.interp(wind_speed, self.Cp_u_wind_speeds, self.Cp_u_values)
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-bpa-tsr":
            cp = 0
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
            raise Exception("Cp calculation based on cp-bpa-tsr not implemented yet.")
        else:
            raise Exception("The power calculation method %s is unkown. Try cp-u lut, cp-bpa-tsr, axial induction "
                            "instead." % self.power_calc_method)

        return p
    

# ====== BART ======

class TurbineSimpleDriveTrain(HAWT_ADM):
    """A turbine model which implements the turbine dynamics (as a simple drive train), including an actual rotor speed and inertia."""

    def __init__(self, base_location, orientation, turbine_states, observation_points, ambient_states, turbine_data, dt, load_model: Literal[ 'first_principles'] | None = None, init_rotor_speed: Optional[float] = None):
        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states, turbine_data)
        #: Add the dynamic states
        # FIXME: These need to be able to be passed to the turbine
        # FIXME: We need to check that all these arguments are here before running this code
        self.azimuth = 0  # In rad
        # FIXME: Right now, these are placeholder values and implementation
        if load_model == 'first_principles':
            self.blade_mass: float = turbine_data['blade_mass']
            self.blade_com: float = turbine_data['blade_com']
            # NOTE: Currently, this is expected to be a lambda expression as a string in the input `.yaml` file, of the form 'lambda fraction: ...'
            self.blade_chord: Callable[[float], float] = eval(turbine_data['blade_chord'])
        # FIXME: This is hard-coded now... should really be passed as an argument
        if init_rotor_speed is not None:
            self.omega = convert(init_rotor_speed, 'RPM', 'rad/s')  # In rad/s
        else:
            self.omega = convert(8, 'RPM', 'rad/s')  # In rad/s
        self.pitch = 0   # In rad
        self.operational_mode: Literal['power_production', 'shutting_down', 'emergency_stop', 'parked', 'starting_up'] = 'power_production'
        self.turbine_data = turbine_data
        self.rotor_radius = self.diameter / 2
        self.inertia = turbine_data['hub_inertia_low_speed_shaft']
        self.generator_efficiency = turbine_data['generator_efficiency']
        self.power_calc_method = 'simple_drive_train'
        self.gen_torque_controller_mode = 'K_omega_squared'  # TODO: Set later in input
        self.dt = dt  # Time step for the dynamics (in seconds)
        #: Calculate the optimal gain K
        # self.K = 1 / (2 * (((turbine_data['performance']['rated_rot_speed'] * self.rotor_radius) / turbine_data['performance']['rated_wind_speed']) ** 3)) * 1.225 * np.pi * (self.rotor_radius ** 5) * turbine_data['performance']['Cp_opt']
        self.Cp_interp = None
        self.pitch_interp = None
        self.Cp_power_mode = 'not_set'
        # FIXME: For now, we have just hard-coded this for the NREL 5MW turbine
        self.K = 2680752.3292693296
        # FIXME: This is just a random constant
        self.brake_torque = 1E6
        self.GRAV_CONST = 9.81  # m/s^2
        self.AIR_DENSITY = 1.225  # kg/m^3
        # FIXME: This should also be defined in the input file
        self.N_BLADE = 3
        
    def calc_power(self, wind_speed, air_den=1.225) -> float:
        """Calculate the power based on turbine, ambient and OP states, and the current turbine dynamics

        Parameters
        ----------
        wind_speed : float
            Wind speed (in m/s)
        air_den : float
            air density (in kg/m^3)

        Returns
        -------
        float :
            Power generated (in W)
        
        """

        def _init_Cp() -> None:
            if "Cp_tb_values" in self.turbine_data["performance"]["Cp_curve"]:
                if isinstance(Cp_tb_values_list := self.turbine_data["performance"]["Cp_curve"]["Cp_tb_values"], list) and isinstance(Cp_tb_values_list[-1], str) and Cp_tb_values_list[-1].endswith('.csv'):
                    #: Convert the list of path segments to a Path
                    Cp_tb_values_path = Path(*Cp_tb_values_list)
                    #: Read the Csv
                    # FIXME: This is now all hardcoded, not really what we want
                    Data = np.loadtxt(Cp_tb_values_path, delimiter=';', skiprows=1)
                    (pitch, tsr, Cp, Ct), stall = [np.flipud(Data[:, i].reshape(300, 120, order='F')) for i in [1, 2, 3, 4]], np.full((300, 120), np.nan)  # NOTE: Pitch is expected in deg
                    # pitch_range, tsr_range = [-10, 50], [0.05, 15]
                    pitch_vals, tsr_vals = np.unique(pitch), np.unique(tsr)
                    # Use only valid table entries for interpolation. The NREL table contains NaN holes,
                    # and RegularGridInterpolator returns NaN whenever a cell corner is missing.
                    Cp_flat = ~np.isnan(Cp).flatten()
                    Cp_points = np.vstack((tsr.flatten()[Cp_flat], pitch.flatten()[Cp_flat])).T
                    Cp_values = Cp.flatten()[Cp_flat]
                    Ct_flat = ~np.isnan(Ct).flatten()
                    Ct_points = np.vstack((tsr.flatten()[Ct_flat], pitch.flatten()[Ct_flat])).T
                    Ct_values = Ct.flatten()[Ct_flat]
                    Cp_interp = sp.interpolate.LinearNDInterpolator(Cp_points, Cp_values, fill_value=np.nan)
                    Ct_interp = sp.interpolate.LinearNDInterpolator(Ct_points, Ct_values, fill_value=np.nan)
                    self.Cp_interp = lambda lbd_pitch, lbd_tsr: float(Cp_interp(np.stack((np.asarray(lbd_tsr), np.asarray(lbd_pitch)), axis=-1))[0])  # NOTE: Pitch is expected in deg
                    self.Ct_interp = lambda lbd_pitch, lbd_tsr: float(Ct_interp(np.stack((np.asarray(lbd_tsr), np.asarray(lbd_pitch)), axis=-1))[0])  # NOTE: Pitch is expected in deg
                    self.Cp_power_mode = 'lookup_table'
                else:
                    raise ValueError("Cp lookup table values should be a list of values or a string ending with .csv")
            else:
                #: Give a warning that blade dynamics are not taken into account
                try: 
                    if not self.warn_raised:
                        warnings.warn(f"No Cp lookup table provided for the turbine. Using a linear interpolation of the Cp curve based on wind speed.", UserWarning, stacklevel=4)
                        self.warn_raised = True
                except AttributeError:
                    self.warn_raised = False
                #: Create an interpolation of the Cp curve based on the wind speed
                Cp_interp = lambda ws: np.interp(ws, self.Cp_u_wind_speeds, self.Cp_u_values)
                self.Cp_interp = Cp_interp
                self.Cp_power_mode = 'wind_speed'

        def _init_pitch() -> None:
            if 'pitch' in self.turbine_data:
                try:
                    pitch_u_values = self.turbine_data['pitch']['pitch_curve']['pitch_u_values']
                    pitch_u_wind_speeds = self.turbine_data['pitch']['pitch_curve']['pitch_u_wind_speeds']
                    self.pitch_interp = lambda ws: np.interp(ws, pitch_u_wind_speeds, pitch_u_values)  # NOTE: Pitch is expected in deg
                except KeyError as e:
                    raise KeyError("Pitch curve should be provided in the input file under the 'pitch' key, with a list of values or a string ending with .csv") from e
            else:
                warnings.warn(f"No pitch curve provided for the turbine. Always using 0 degrees.", UserWarning, stacklevel=4)
                self.pitch_interp = lambda ws: 0

        #: Check if the turbine is active
        if self.operational_mode == 'parked':
            if self.omega != 0:
                warnings.warn("Turbine is parked but rotor speed is not zero. Setting power and rotor speed to zero, but this should normally not happen.", UserWarning, stacklevel=4)
                self.omega = 0
            return 0
        elif self.operational_mode == 'starting_up':
            if self.omega == 0:
                self.omega = convert(0.1, 'RPM', 'rad/s')  # FIXME: This is just a placeholder value, should be set based on the turbine data and startup procedure
        
        #: Extract the Cp curve
        if self.Cp_interp is None:
            _init_Cp()

        #: Extract the pitch curve
        if self.pitch_interp is None:
            _init_pitch()

        if self.yaw_power_coeff == "pP":
            yaw = np.deg2rad(self.turbine_states.get_current_yaw())
            yaw_coef = np.cos(yaw) ** self.Cp_pP
        elif self.yaw_power_coeff == "none":
            yaw_coef = 1.0
        else:
            raise Exception("Only cos(yaw) ** pP yaw coefficient supported (or none)")
        
        if self.power_calc_method == "axial induction":
            axi = self.turbine_states.get_current_ax_ind()
            cp = 4 * axi * (1 - axi) ** 2
            power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-u lut":
            cp = np.interp(wind_speed, self.Cp_u_wind_speeds, self.Cp_u_values)
            power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-bpa-tsr":
            cp = 0
            power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
            raise Exception("Cp calculation based on cp-bpa-tsr not implemented yet.")
        elif self.power_calc_method == 'simple_drive_train':

            #: Calculate the tip speed ratio
            tsr_t = (self.omega * self.rotor_radius) / wind_speed if wind_speed > 0 else 0

            #: Calculate the pitch
            # FIXME: This gives really funky results in above-rated conditions; everything goes to zero
            if self.Cp_power_mode == 'wind_speed':
                try: 
                    if not self.Cp_warn_raised:
                        warnings.warn(f"Pitch is not taken into account in the Cp calculation (Cp_power_mode={self.Cp_power_mode}). This may lead to inconsistent results.", UserWarning, stacklevel=4)
                        self.Cp_warn_raised = True
                except AttributeError:
                    self.Cp_warn_raised = False
            if wind_speed < self.turbine_data['performance']['rated_wind_speed']:
                self.pitch = 0
            else:
                self.pitch = np.deg2rad(self.pitch_interp(wind_speed))
            if self.operational_mode in ['shutting_down', 'emergency_stop']:
                self.pitch = np.deg2rad(30.5)  # FIXME: Should be 90, but without actuator dynamics this seems to 'crash' the turbine | FIXME: Now, Cp is not becoming zero....

            #: Calculate the Cp coefficient
            match self.Cp_power_mode:
                case 'wind_speed':
                    args = (wind_speed,)
                case 'lookup_table':
                    args = (np.rad2deg(self.pitch), tsr_t)
                case _:
                    raise ValueError(f"Unsupported Cp power mode '{self.Cp_power_mode}'")
            Cp_t = self.Cp_interp(*args)

            #: Calculate the aerodynamic torque
            # FIXME: This results in error if the rotor speed is zero
            if np.isclose(self.omega, 0):
                warnings.warn("Rotor speed is zero, setting aerodynamic torque to 1E4 Nm to avoid division by zero", RuntimeWarning, stacklevel=4)
                # T_a = 1E4  # FIXME: Why this value?
                T_a = 1E4  # FIXME: Why this value?
            else:
                T_a = 1 / (2 * self.omega) * air_den * np.pi * (self.rotor_radius ** 2) * Cp_t * (wind_speed ** 3)

            #: Calculate the generator torque
            # FIXME: For now, the controller mode is hardcoded here, but it should be passed as an argument
            match self.gen_torque_controller_mode:
                case 'K_omega_squared':
                    T_g = self.K * (self.omega ** 2)
                case _:
                    raise ValueError(f"Unsupported controller mode '{self.gen_torque_controller_mode}'")
            if self.operational_mode == 'emergency_stop':
                T_g = 0  # FIXME: Is this really accurate?

            #: Check if there are any shutdown events
            match self.operational_mode:
                case 'power_production':
                    T_brake = 0
                case 'shutting_down':
                    if self.omega == 0:
                        self.operational_mode = 'parked'
                        return 0
                    T_brake = 0
                case 'emergency_stop':
                    if self.omega == 0:
                        self.operational_mode = 'parked'
                        return 0
                    T_brake = self.brake_torque
                    # # FIXME: This is kind of a weird hack... but makes the stopping more 'smooth'
                    # T_a = np.tanh(0.1 * self.omega) * T_a
                case 'parked':
                    pass  # NOTE: Should have been caught at the top of this method
                case 'starting_up':
                    if self.omega >= convert(1, 'RPM', 'rad/s'):  # FIXME: This is just a placeholder value, should be set based on the turbine data and startup procedure
                        self.operational_mode = 'power_production'
                    T_brake = 0
                case _:
                    raise ValueError(f"Unsupported operational mode '{self.operational_mode}'")

            #: Calculate the rotor acceleration
            omega_dot_t = (T_a - T_g - T_brake) / self.inertia
            #: Calculate the power output
            power = T_g * self.generator_efficiency * self.omega * yaw_coef  # Power output in Watts
            #: Calculate the new rotor speed
            self.omega = self.omega + omega_dot_t * self.dt
            #: Make sure the rotor speed does not go below zero
            # FIXME: Is this really a good way to handle this?
            self.omega = max(0, self.omega)
            #: Calculate the new azimuth angle
            # FIXME: Do we actually wan't to update that here? Probably not, right? Maybe just at either the very beginning, the first function call, or the very end one?
            self.azimuth += self.omega * self.dt
            self.azimuth %= 2 * np.pi  # Keep the azimuth angle between 0 and 2pi
        else:
            raise Exception(f"The power calculation method {self.power_calc_method} is unkown. Try cp-u lut, cp-bpa-tsr, axial induction instead.")
        return power
    
    def calc_loads(self, vis_tile: Callable[[tuple[float, float, float]], float], int_mode: Literal['scipy', 'riemann'] = 'riemann', nint_points: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the blade root flapwise bending moment, edgewise bending moment, and normal force."""

        def get_blade_coords(blade_azimuth: float,fraction: float) -> np.ndarray:
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
            # NOTE: This function assumes `blade_azimuth` is defined in its outer scope

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

            #: Calculate local blade coordinates around the y-axis
            # NOTE: It seems to be that, within FLORIS, the yaw angles is NOT defined as being positive clockwise; as such, we need to subtract that angle here in order to make the right calculation
            # FROM: https://nrel.github.io/floris/examples/examples_control_optimization/001_opt_yaw_single_ws.html  # nopep8
            x_local = rot_mat_y(blade_azimuth) @ np.array([0, 0, fraction * self.rotor_radius])
            # Calculate the total rotation around the z-axis
            # NOTE: In this model, we are ignoring the tilt angle
            # FIXME: Here, something seems to be wrong; when the turbine is yawed w.r.t. the wind direction, this should still work, as we are taking the global orientation of the wind direction; however, it seems the flow sampler is still 'expecting' the turbine to be oriented w.r.t. the wind direction, and the wind speeds that we get are NOT correct. NO, it turns out the issue is deeper; even selecting both to be 260 degrees, the methodology does not seem to work. So I guess it's really the rotations which are off, and we need to investigate that further. When does it work: wd 270, wt 270, NOT work: wd 260, wt 260. Wait! If wd 260, hardcoding wt 280 seems to work... Yes! wd 200, hardcoding wt 270 + 70, but indeed, wd 200, with wt 210, but hardcoding 270 + 70 (so based on wd) does work; however, wd 200, wt 210, but hardcoding 270 + 60 does NOT work. Si it seems 270 + (270 - wd)
            total_turbine_orientation_rad = np.radians(180 - self.ambient_states.get_wind_dir()[0])
            x_global = rot_mat_z(total_turbine_orientation_rad) @ x_local
            return x_global
        
        def dist_wind_load(r: float, blade_azimuth: float) -> float:
            """Calculate the distributed wind load (in N/m) at a given radial position along the blade.

            Parameters
            ----------
            r : float
                Radial position along the blade (in m)

            Returns
            -------
            float:
                Distributed wind load at the given radial position (in N/m)
            """
            # NOTE: This function assumes `blade_azimuth` is defined in its outer scope 
            
            #: Normalize the radial position
            fraction = r / self.rotor_radius  # NOTE: This is a value between 0 and 1
            #: Get the global coordinates of the radial position based on wind direction, turbine orientation, and azimuth angle
            coords = get_blade_coords(blade_azimuth, fraction) + self.get_rotor_pos()
            #: Calculate the local wind speed at the given radial position
            local_wind_speed = vis_tile(np.atleast_1d(coords[0]), np.atleast_1d(coords[1]), np.atleast_1d(coords[2]))
            #: Calculate the local chord length at the given radial position
            local_chord = self.blade_chord(r)
            #: Calculate the distributed wind load
            # FIXME: This is actually not correct when sp.integrate.quad is used, as it already integrates over `dr`, so we should not multiply by `local_chord * (self.rotor_radius / nint_points)` here, and nint_points is NOT the number of points used in the integration!
            dist_load = 0.5 * self.AIR_DENSITY * local_wind_speed ** 2 * (local_chord * (self.rotor_radius / nint_points))  # in N/m
            #: Return the result
            return dist_load
        
        #: Initialize the results
        flapwise_bending_moment, edgewise_bending_moment, normal_force = [np.zeros(self.N_BLADE) for _ in range(3)]
        #: Loop over all the blades
        for blade_idx in range(self.N_BLADE):
            #: Calculate the azimuth angle of the blade
            blade_azimuth = self.azimuth + np.radians(blade_idx * 120)
            #: Calculate the centrifugal force
            centrifugal_force = self.blade_mass * (self.omega ** 2) * self.rotor_radius
            #: Calculate the gravitational force
            gravitational_force_par, gravitational_force_perp = self.blade_mass * self.GRAV_CONST * (-1) * np.cos(blade_azimuth), self.blade_mass * self.GRAV_CONST * (-1) * np.sin(blade_azimuth)  # NOTE: This makes sure that 0 deg is fully pointing negative in the parallel blade direction (x-axis), and 90 deg is fully pointing negative in the perpendicular blade direction (y-axis)
            #: Calculate the normal force
            normal_force[blade_idx] = centrifugal_force + gravitational_force_par
            #: Calculate the edgewise bending moment
            # TODO: Here, this should actually also include the lift component, as this bends the blade in the edgewise direction as well
            edgewise_bending_moment[blade_idx] = gravitational_force_perp * self.blade_com  # NOTE: Here, we assume that clockwise rotation is positive, so a bending moment 'downwards' is positive
            #: Calculate the flapwise bending moment
            # NOTE: Here, we assume that 'backwards' (in the same direction as the wind) bending is positive
            # TODO: Also incorporate the angle at which the wind hits the blade, both due to wind direction + yaw, but also a loss factor due to the blades pitching. Note that this also induces a edgewise component (misaligned wind direction)!
            #: Match the mode of calculating the moment
            # FIXME: The loads, very clearly, do NOT seem to be correct, because there are very sharp discontinuous jumps in the load time series. Need to investigate this further. 
            match int_mode:
                case 'scipy':
                    flapwise_bending_moment[blade_idx], *_ = sp.integrate.quad(lambda r, blade_azimuth: dist_wind_load(r, blade_azimuth) * r, 0, self.rotor_radius, args=(blade_azimuth,))
                case 'riemann':
                    dr = self.rotor_radius / nint_points
                    r_values = np.linspace(dr / 2, self.rotor_radius - dr / 2, nint_points)  # Midpoint Riemann sum
                    #: Calculate the coordinate values
                    coords = np.zeros((3, nint_points))
                    for idx in range(nint_points):
                        coords[:, idx] = get_blade_coords(blade_azimuth, idx / nint_points) + self.get_rotor_pos()
                    #: Calculate the local wind speeds at all radial positions
                    local_wind_speeds = vis_tile(coords[0, :], coords[1, :], coords[2, :]).squeeze()
                    #: Calculate the distributed wind loads at all radial positions
                    # TODO: Add the blade-pitch induced loss factor here, with some sine-cosine, due to the angling of the blade. Also, this could be a fully-fledged drag-coefficient based calculation, instead of just a flat distributed load.
                    distributed_wind_loads = 0.5 * self.AIR_DENSITY * local_wind_speeds ** 2 * (self.blade_chord(np.linspace(0, 1, nint_points)) * (self.rotor_radius / nint_points))  # in N/m
                    #: Calculate the  flapwise bending moment
                    flapwise_bending_moment[blade_idx] = np.sum([distributed_wind_loads[idx] * r_values[idx] * dr for idx in range(nint_points)])
                case _:
                    raise ValueError(f"Unsupported integration mode '{int_mode}'")
        #: Return the results
        return flapwise_bending_moment, edgewise_bending_moment, normal_force
        

# FIXME: I don't know if this is the best/correct way to implement the downregulation wind turbine
class TurbineSimpleDriveTrainDownregulation(TurbineSimpleDriveTrain):
    """A simple drive train turbine model for downwind turbines, which extends the TurbineSimpleDriveTrain model."""

    def __init__(self, base_location, orientation, turbine_states, observation_points, ambient_states, turbine_data, dt, load_model: Literal[ 'first_principles'] | None = None, init_rotor_speed: Optional[float] = None):
        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states, turbine_data, dt, load_model, init_rotor_speed)
        # FIXME: This is also a temporary overwrite, no idea if its correct
        self.inertia: float = float(turbine_data['inertia'])
        self.rated_power: float = turbine_data['performance']['rated_power']  # in W
        self.rated_rotor_speed: float = turbine_data['performance']['rated_rotor_speed']  # in RPM
        self.opt_Cp: float = 0.482  # FIXME: Currently hardcoded, but should be passed as an argument
        self.opt_tsr: float = turbine_data['TSR']  # Optimal tip speed ratio for power production
        # FIXME: Right now, we load this externally; of course, not what we want to do
        #
        with open('02_Examples_and_Cases/00_Inputs/00_OFF/05_Turbine/NREL5MW/Cp_Ct_NREL5MW_nrel.csv', 'r') as f:
            # Load full matrix for 2D interpolation
            full_data = np.loadtxt(f, delimiter=';', skiprows=1)
            # You'll need to adjust these indices based on your actual CSV structure
            self.tsr_values = np.unique(full_data[:, 2])
            self.pitch_values = np.unique(full_data[:, 1])
            # Reshape Cp values into 2D matrix (n_tsr x n_pitch)
            self.Cp_matrix = full_data[:, 3].reshape(len(self.pitch_values), len(self.tsr_values)).T
        #
        self.T_g: float = 1E6  # Generator torque, initialized to 1E6
        self.pitch: float = 0  # Blade pitch angle, initialized to zero
        self.power_setpoint: Optional[float] = None  # Power setpoint for downregulation, initialized to None (no downregulation)

    def calc_power(self, wind_speed, air_den=1.225) -> float:
        """Calculate the power based on turbine, ambient and OP states, and the current turbine dynamics

        Parameters
        ----------
        wind_speed : float
            Wind speed (in m/s)
        air_den : float
            air density (in kg/m^3)

        Returns
        -------
        float :
            Power generated (in W)
        
        """
        #: Check if downregulation is active
        if self.power_setpoint is None:
            return super().calc_power(wind_speed, air_den)
        else:
            #: Extract the Cp curve
            if "Cp_tb_values" in self.turbine_data["performance"]["Cp_curve"]:
                raise NotImplementedError("Cp calculation based on Cp lookup table not implemented yet")
            else:
                #: Give a warning that blade dynamics are not taken into account
                try: 
                    if not self.warn_raised:
                        warnings.warn(f"No Cp lookup table provided for the turbine. Using a linear interpolation of the Cp curve based on wind speed.", UserWarning, stacklevel=4)
                        self.warn_raised = True
                except AttributeError:
                    self.warn_raised = False
                #: Create an interpolation of the Cp curve based on the wind speed
                # NOTE: This MUST be set to 'lookup_table', otherwise the blade pitch will not have any effect
                Cp_power_mode = 'lookup_table'
                from scipy.interpolate import LinearNDInterpolator
                Cp_flat = ~np.isnan(self.Cp_matrix).flatten()
                Cp_points = np.vstack((np.repeat(self.tsr_values, len(self.pitch_values))[Cp_flat], np.tile(self.pitch_values, len(self.tsr_values))[Cp_flat])).T
                Cp_values = self.Cp_matrix.flatten()[Cp_flat]
                _Cp_interp = LinearNDInterpolator(Cp_points, Cp_values, fill_value=np.nan)
                # Create a wrapper function for easier calling with proper input formatting
                Cp_interp = lambda tsr, pitch: _Cp_interp(np.array([[float(tsr), float(pitch)]]))[0]

            if self.yaw_power_coeff == "pP":
                yaw = np.deg2rad(self.turbine_states.get_current_yaw())
                yaw_coef = np.cos(yaw) ** self.Cp_pP
            elif self.yaw_power_coeff == "none":
                yaw_coef = 1.0
            else:
                raise Exception("Only cos(yaw) ** pP yaw coefficient supported (or none)")
            
            if self.power_calc_method == "axial induction":
                axi = self.turbine_states.get_current_ax_ind()
                cp = 4 * axi * (1 - axi) ** 2
                power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
            elif self.power_calc_method == "cp-u lut":
                cp = np.interp(wind_speed, self.Cp_u_wind_speeds, self.Cp_u_values)
                power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
            elif self.power_calc_method == "cp-bpa-tsr":
                cp = 0
                power = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
                raise Exception("Cp calculation based on cp-bpa-tsr not implemented yet.")
            elif self.power_calc_method == 'simple_drive_train':
                #: Calculate the tip speed ratio
                tsr_t = (self.omega * self.rotor_radius) / wind_speed if wind_speed > 0 else 0
                #: Calculate the Cp coefficient
                match Cp_power_mode:
                    case 'wind_speed':
                        args = (wind_speed,)
                    case 'lookup_table':
                        args = (tsr_t, np.rad2deg(self.pitch))  # NOTE: Expect the pitch to be in degrees
                    case _:
                        raise ValueError(f"Unsupported Cp power mode '{Cp_power_mode}'")
                Cp_t = Cp_interp(*args)
                #: Calculate the aerodynamic torque
                # FIXME: Sometimes this can be np.nan, which is problematic...
                if np.isnan(Cp_t):
                    warnings.warn(f"Cp value is NaN, tsr {tsr_t:.2f}, pitch {convert(self.pitch, 'rad', 'deg'):.2f}, using 0 instead")
                    Cp_t = 0
                T_a = 1 / (2 * self.omega) * self.AIR_DENSITY * np.pi * (self.rotor_radius ** 2) * Cp_t * (wind_speed ** 3)
                #: Calculate the rotor acceleration
                omega_dot_t = (T_a - self.T_g) / self.inertia
                #: Calculate the power output
                power = self.T_g * self.generator_efficiency * self.omega  # Power output in Watts
                #: Calculate the new rotor speed
                self.omega = self.omega + omega_dot_t * self.dt
                #: Calculate the new azimuth angle
                # FIXME: Do we actually wan't to update that here? Probably not, right? Maybe just at either the very beginning, the first function call, or the very end one?
                self.azimuth += self.omega * self.dt
                self.azimuth %= 2 * np.pi  # Keep the azimuth angle between 0 and 2pi
        return power
        
# ====== BART ======


class TurbineStatesFLORIDyn(TurbineStates):

    def __init__(self, number_of_time_steps: int):
        """
        TurbineStatesFLORIDyn includes the axial induction factor, the yaw misalignment and the added turbulence
        intensity.

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        """
        super().__init__(number_of_time_steps, 3, ['axial induction (-), yaw (deg), added turbulence intensity (%)'])

    def get_current_cp(self) -> float:  # TODO: Remove! This has been moved to the turbine model
        """
        get_current_cp returns the current power coefficient of the turbine

        Returns
        -------
        float:
            Power coefficient (-)
        """
        return 4 * self.states[0, 0] * (1 - self.states[0, 0]) ** 2 * \
               np.cos(self.states[0, 1]) ** 2.2  # TODO Double check correct Cp calculation

    def get_current_ct(self) -> float:  # TODO: Remove! This has been moved to the turbine model
        """
        get_current_ct returns the current thrust coefficient of the turbine

        Returns
        -------
        float:
            Thrust coefficient (-)
        """
        return self.get_ct(0)

    def get_current_ax_ind(self) -> float:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        if self.n_time_steps > 1:
            return self.states[0, 0]
        else:
            return self.states[0]

    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location

        Returns
        -------
        float:
            yaw misalignment at the turbine location (deg)
        """
        return self.get_yaw(0)

    def set_yaw(self, yaw_angle: float):
        """
        Sets the yaw misalignment with the wind direction

        Parameters
        ----------
        yaw_angle
        """
        if self.n_time_steps > 1:
            self.states[0, 1] = yaw_angle
        else:
            self.states[1] = yaw_angle

    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which Ct should be calculated
        Returns
        -------
        float:
            Thrust coefficient
        """
        ax_i = self.get_ax_ind(index)
        yaw = np.deg2rad(self.get_yaw(index))
        return 4 * ax_i * (1 - ax_i) * \
               np.cos(yaw) ** 2.2  # TODO Insert correct Ct calculation

    def get_ax_ind(self, index: int) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        if self.n_time_steps > 1:
            return self.states[index, 0]
        else:
            return self.states[0]

    def set_ax_ind(self, ax_ind):
        """
        Stores the axial induction factor of the turbine in the states.

        Parameters
        ----------
        ax_ind: axial induction factor
        """
        if self.n_time_steps > 1:
            self.states[0, 0] = ax_ind
        else:
            self.states[0] = ax_ind

    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index

        Parameters
        ----------
        index : int

        Returns
        -------
        float:
            yaw misalignment in deg
        """
        if self.n_time_steps > 1:
            return self.states[index, 1]
        else:
            return self.states[1]

    def get_all_ct(self) -> np.ndarray:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.

        Returns
        -------
        np.ndarray:
            Thrust coefficient at all turbine states (-)
        """
        # TODO vectorized calculation of Ct
        pass

    def get_all_ax_ind(self) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        return self.states[:, 1]

    def get_all_yaw(self) -> np.ndarray:
        """
        returns the yaw misalignment for all turbine states.

        Returns
        -------
        np.ndarray:
            n x 1 vector with all yaw angles
        """
        return self.states[:, 1]

    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates a TurbineStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_current_yaw()

        Parameters
        ----------
        index1 : int
            Index of the first state
        index2 : int
            Index of the second state
        w1 : float
            Weight for first index (has to be w1 = 1 - w2, and [0,1])
        w2 : float
            Weight for second index (has to be w2 = 1 - w1, and [0,1])

        Returns
        -------
        TurbineStates
            turbine state object with single entry
        """
        # TODO create check for weights
        t_s = TurbineStatesFLORIDyn(1)
        t_s.set_all_states(self.states[index1, :]*w1 + self.states[index2, :]*w2)
        return t_s


# SOURCES
# [1] The Dtu 10-Mw Reference Wind Turbine, Bak et al., 2013
#       https://orbit.dtu.dk/en/publications/the-dtu-10-mw-reference-wind-turbine
