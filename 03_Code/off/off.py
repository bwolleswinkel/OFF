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

import os

import logging
lg = logging.getLogger('off')

import off.windfarm as wfm
import off.ambient_corrector as amb_corr
import off.utils as util
import off.controller as ctr
import off.vis_flow_field as vff
import numpy as np
import pandas as pd
import off.wake_solver as ws
from off.logger import CONSOLE_LVL, FILE_LVL, Formatter, _logger_add
import shutil

# ====== BART ======
from off.turbine import TurbineSimpleDriveTrain, TurbineSimpleDriveTrainDownregulation
# ====== BART ======

from off import __file__ as OFF_PATH
import datetime

OFF_PATH = OFF_PATH.rsplit('/', 3)[0]


class OFF:
    """
    OFF is the central object which initializes the wind farm and runs the simulation
    """
    settings_sim = dict()
    wind_farm = wfm.WindFarm
    settings_vis = dict()

    def __init__(self, wind_farm: wfm.WindFarm, settings_sim: dict, settings_wke: dict, settings_sol: dict,
                 settings_cor: dict, settings_ctr: dict, settings_ctr_all: dict, settings_turbine: dict, settings_events: dict, vis: dict):
        self.wind_farm = wind_farm
        self.settings_sim = settings_sim
        self.settings_vis = vis

        # ====== BART ======
        self.settings_cor = settings_cor
        self.settings_turbine = settings_turbine
        self.settings_events = settings_events
        # ====== BART ======

        self.__dir_init__( settings_sim )
        self.__logger_init__( settings_sim )
        settings_wke['sim_dir'] = self.root_dir

        # =========== FLORIS ===========
        settings_wke['yaml_path'] = self.sim_dir + '/FLORIS.yaml'
        shutil.move(settings_wke['tmp_yaml_path'], settings_wke['yaml_path'])

        # =========== Solver ===========
        # self.wake_solver = ws.FLORIDynTWFWakeSolver(settings_wke, settings_sol)
        # self.wake_solver = ws.FLORIDynFlorisWakeSolver(settings_wke, settings_sol)
        self.wake_solver = ws.TWFSolver(settings_wke, settings_sol, vis)

        # ====== BART ======
        # ====== WIND TURBINE DYNAMICS ======
        try:
            self.wt_dynamics_model = settings_turbine['dynamics']['model']
        except KeyError:
            pass
        # ====== BART ======

        # =========== Controller ===========
        if settings_ctr["ctl"] == "IdealGreedyBaseline":
            self.controller = ctr.IdealGreedyBaselineController(settings_ctr)
        elif settings_ctr["ctl"] == "RealGreedyBaseline":
            self.controller = ctr.RealisticGreedyBaselineController(settings_ctr)
        elif settings_ctr["ctl"] == "prescribed filtered yaw controller":
            self.controller = ctr.YawSteeringFilteredPrescribedMotionController(settings_ctr)
        elif settings_ctr["ctl"] == "prescribed yaw controller":
            self.controller = ctr.YawSteeringPrescribedMotionController(settings_ctr)
        elif settings_ctr["ctl"] == "LUT yaw controller":
            self.controller = ctr.YawSteeringLUTController(settings_ctr)
        elif settings_ctr["ctl"] == "Dead-band LUT yaw controller":
            self.controller = ctr.DeadbandYawSteeringLuTController(settings_ctr, self.settings_sim['time step'], self.wind_farm.nT)
        else:
            raise Warning("Controller %s is undefined!" % settings_ctr["ctl"])
        
        # ====== BART ======
        # ====== POWER CONTROLLER ======
        if 'power_controller' in settings_ctr_all:
            match settings_ctr_all['power_controller']['settings']['type']:
                case 'lio':
                    self.power_controller = ctr.DownregulationControllerLio(settings_ctr_all['power_controller']['settings']['strategy'], {'power_factor': settings_ctr_all['power_controller']['settings']['power_factor'], 'power_t': settings_ctr_all['power_controller']['settings']['power_t']})
                case _:
                    raise ValueError(f"Power controller {settings_ctr_all['power_controller']['settings']['type']} is not recognized.")
        else:
            self.power_controller = None
        # ====== BART ======

        # =========== Corrector ===========
        if settings_cor['ambient']: 
            states_name = self.wind_farm.turbines[0].ambient_states.get_state_names()
            self.ambient_corrector = amb_corr.AmbientCorrector(settings_cor['ambient'], self.wind_farm, states_name)

        # =========== Visualization ===========
        self.visualizer_ff = vff.Visualizer_FlowField(self.settings_vis, wind_farm.get_layout()[:,:2])

        # =========== Progess Bar ===========
        self.iterations_total = int((self.settings_sim['time end'] - self.settings_sim['time start']) / self.settings_sim['time step'])

    def __get_runid__(self) -> int:        
        """ Extract and increment the run id

        Returns
        -------
        int
            Current run id.
        """
        current_time = datetime.datetime.now()
        integer = int(current_time.strftime("%Y%m%d%H%M%S%f"))

        # ====== BART ======
        # Add spacing to the run id
        run_id_str = str(integer)
        run_id_str = 'D' + run_id_str[:4] + '_' + run_id_str[4:6] + '_' + run_id_str[6:8] + '_T' + run_id_str[8:10] + '_' + run_id_str[10:12] + '_' + run_id_str[12:14]
        # ====== BART ======

        return integer, run_id_str

    def __dir_init__(self, settings_sim: dict):
        """ Initialize the simulation folder and set the data path.

        Parameters
        ----------
        settings_sim : dict
            Dictionary containing the OFF parameters.

            Available options:
            
            :simulation folder:  *(str)*       - 
                Path to the folder where the simulation results and logs will be
                exported.Export directory name where figures and data are saved. 
                If ``''``, all exports are disabled; if None (by default), default 
                export directory name ``off_run_id``.
            :data folder:        *(str)*       - 
                Path to the folder containing the off data.
        """

        sim_dir  = settings_sim.setdefault('simulation folder', None)
        data_dir = settings_sim.get('data folder', None)

        run_id, run_id_spaced = self.__get_runid__()

        try:
            root_dir = data_dir or f'{os.environ["OFF_PATH"]}/runs/'
            lg.info('Root runs directory: ' + root_dir)
        except KeyError:
            if os.environ["PWD"].endswith("03_Code"):
                root_dir = data_dir or f'{os.environ["PWD"][:-len("03_Code")]}/runs/'
            else:
                root_dir = data_dir or f'{os.environ["PWD"]}/runs/'
            lg.warning('Initial root runs directory path retrieval was unsuccessful, used ' + root_dir)

        self.sim_dir = f'{root_dir}/off_run_{run_id_spaced}' if sim_dir is None else sim_dir
        self.root_dir = root_dir[:-len("runs/")]

        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
            lg.info('Created simulation directory at ' + self.sim_dir)

    def __logger_init__(self, settings_sim: dict):
        """ Initializes the logger

        Parameters
        ----------
        settings_sim : dict
            Dictionary containing the OFF parameters
            
            :log console lvl:  *(str)*                  - 
                Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``) used 
                for the console logging.
            :log file lvl:     *(str)*                  - 
                Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``) used 
                for the file logging. Log logged to ``self.sim_dir/off.log``. 
                Not available if ``self.sim_dir`` not set.
        """
        file_lvl = settings_sim.setdefault('log file lvl', FILE_LVL).upper()
        console_lvl = settings_sim.setdefault('log console lvl', CONSOLE_LVL).upper()

        min_lvl = min(getattr(logging, file_lvl), getattr(logging, console_lvl))
        lg.setLevel(min_lvl)
        lg.propagate = False

        file_formatter = Formatter('%(levelname)s : %(filename)s, line %(lineno)d in %(funcName)s : %(message)s')
        console_formatter = Formatter('%(levelname)s : %(message)s')

        _logger_add(lg, logging.StreamHandler(), console_lvl, console_formatter)
        if self.sim_dir:
            if not self.sim_dir:
                lg.warning('Not simulation folder was specified: file logger disabled.')
            else: 
                log_fid = f'{self.sim_dir}/off.log'
                _logger_add(lg, logging.FileHandler(log_fid), file_lvl, file_formatter)
        
        lg.info('Saving data to %s.' % self.sim_dir)

    def init_sim(self, start_ambient: np.ndarray, start_turbine: np.ndarray):
        """
        Function which initializes the states within the ``self.wind_farm`` object. 
        Assigns turbine & ambient states and distributes the OPs downstream. OP 
        locations are not necessarily correct but the wakes are "unrolled" and 
        do not have to first develop.

        Parameters
        ----------
        start_ambient : np.ndarray
            1 x n vector with initial ambient state
        start_turbine : np.ndarray
            1 x n vector with initial turbine state
        """        

        for t in self.wind_farm.turbines:
            t.ambient_states.init_all_states(start_ambient)
            t.turbine_states.init_all_states(start_turbine)
            t.observation_points.init_all_states(t.ambient_states.get_turbine_wind_speed_u(),
                                                 t.ambient_states.get_turbine_wind_speed_v(),
                                                 t.get_rotor_pos(), self.settings_sim['time step'])
            pass

    def run_sim(self) -> tuple:
        """
        Central function which executes the simulation and manipulates the ``self.wind_farm object``

        Returns
        -------
        pandas.Dataframe :
            Measurements from the entire simulation
        """
        lg.info(f'Running simulation from {self.settings_sim["time start"]} s to {self.settings_sim["time end"]} s.')
        lg.info(f'Time step: {self.settings_sim["time step"]} s.')

        # Allocate data structures for measurement (output), effective rotor wind speed (u,v) as well as the power
        measurements = pd.DataFrame()
        control_applied = pd.DataFrame()

        uv_r = np.zeros((len(self.wind_farm.turbines), 2))
        pow_t = np.zeros((len(self.wind_farm.turbines), 1))
        
        # ====== BART ======
        # FIXME: Now, the number of blades is still hardcoded to 3
        flapwise_bending_moment_t = np.zeros((3, len(self.wind_farm.turbines)))
        edgewise_bending_moment_t = np.zeros((3, len(self.wind_farm.turbines)))
        normal_force_t = np.zeros((3, len(self.wind_farm.turbines)))
        # ====== BART ======

        iteration = 0

        for t in np.arange(self.settings_sim['time start'],
                           self.settings_sim['time end'],
                           self.settings_sim['time step']):
            lg.info('Starting time step: %s s.' % t)

            # ///////////////////// PREDICT ///////////////////////
            # Get wind speeds at the rotor plane and to propagate the OPs
            for turb_idx, tur in enumerate(self.wind_farm.turbines):
                # Debug flags
                if (self.settings_vis["debug"]["effective_wf_layout"] and
                        t in self.settings_vis["debug"]["time"] and
                        turb_idx in self.settings_vis["debug"]["iT"]):
                    # Plots the wind farm as simulated in the steady state model
                    self.wake_solver.raise_flag_plot_wakes(t)

                if (self.settings_vis["debug"]["effective_wf_tile"] and
                        t in self.settings_vis["debug"]["time"]):
                    # Set flag to calculate wind speed in wake model at grid points belonging to turbine iT
                    grid_points_iT = self.visualizer_ff.vis_get_grid_points_iT(turb_idx)
                    self.wake_solver.raise_flag_plot_tile(
                        grid_points_iT[:,0], grid_points_iT[:,1],
                        np.array(self.settings_vis["grid"]["slice_2d_xy"]))

                # for turbine 'tur': Run wake solver and retrieve measurements from the wake model
                uv_r[turb_idx, :], uv_op, m_tmp = self.wake_solver.get_measurements(turb_idx, self.wind_farm)

                # ====== BART ======
                # Update operational mode based on events
                if self.settings_events is not None:
                    if self.settings_events["settings"]["event_type"] == "shutdown":
                        for event_idx, event_t in enumerate(self.settings_events["settings"]["shutdown_t"]):
                            if t >= event_t and turb_idx in self.settings_events["settings"]["shutdown_indices"][event_idx]:
                                if tur.operational_mode == 'power_production':
                                    tur.operational_mode = 'shutting_down'
                else:
                    pass
                # ====== BART ======

                # Calculate the power generated
                pow_t[turb_idx, :] = tur.calc_power(util.ot_uv2abs(uv_r[turb_idx, 0], uv_r[turb_idx, 1]))
                m_tmp['power_OFF'] = pow_t[turb_idx, :]

                # ====== BART ======
                #: Calculate the loads on the turbine
                if 'dynamics' in self.settings_turbine and 'loads' in self.settings_turbine['dynamics'] and self.settings_turbine['dynamics']['loads'] == 'first_principles':
                    flapwise_bending_moment_t[:, turb_idx], edgewise_bending_moment_t[:, turb_idx], normal_force_t[:, turb_idx] = tur.calc_loads(self.wake_solver.floris_wake.vis_tile)
                else:
                    pass
                # ====== BART ======

                # Add turbine index & timestamp to data
                m_tmp.t_idx = turb_idx
                m_tmp['time'] = t

                # ====== BART ======
                # Save the operational mode in the measurements
                if isinstance(tur, TurbineSimpleDriveTrain):
                    m_tmp['operational_mode'] = tur.operational_mode
                # Save the rotor speed in the measurements
                if isinstance(tur, TurbineSimpleDriveTrain):
                    m_tmp['omega'] = tur.omega
                # Save the blade azimuth in the measurements
                if isinstance(tur, TurbineSimpleDriveTrain):
                    for blade_idx in range(3):
                        m_tmp[f'blade_{blade_idx + 1}_azimuth'] = tur.azimuth + blade_idx * 2 * np.pi / 3
                # Add loads to data
                if 'dynamics' in self.settings_turbine and 'loads' in self.settings_turbine['dynamics'] and self.settings_turbine['dynamics']['loads'] == 'first_principles':
                    for blade_idx in range(flapwise_bending_moment_t.shape[0]):
                        m_tmp[f'flapwise_bending_moment_blade_{blade_idx + 1}'] = flapwise_bending_moment_t[blade_idx, turb_idx]
                        m_tmp[f'edgewise_bending_moment_blade_{blade_idx + 1}'] = edgewise_bending_moment_t[blade_idx, turb_idx]
                        m_tmp[f'normal_force_blade_{blade_idx + 1}'] = normal_force_t[blade_idx, turb_idx]
                else:
                    pass
                # ====== BART ======

                # ====== BART ======
                # Save the rotor speed and rotor speed setpoint in the measurements
                if isinstance(tur, TurbineSimpleDriveTrainDownregulation):
                    # Save the rotor speed setpoint
                    m_tmp['omega_setpoint'] = self.power_controller.rotor_setpoint[turb_idx]
                    # Save the power setpoint in the measurements
                    m_tmp['power_setpoint'] = self.power_controller.power_setpoint[turb_idx]
                    # Save the derated wind speed in the measurements
                    m_tmp['wind_speed_derated'] = self.power_controller.wind_speed_derated[turb_idx]
                    # Save the generator torque setpoint in the measurements
                    m_tmp['T_g'] = self.power_controller.T_g[turb_idx]
                    # Save the pitch angle setpoint in the measurements
                    m_tmp['pitch'] = np.rad2deg(self.power_controller.pitch[turb_idx])  # In degrees
                # ====== BART ======

                # Append turbine measurements to general measurement data
                measurements = pd.concat([measurements, m_tmp], ignore_index=True)

                # Set propagation speed of the OPs of the turbine 'tur'
                tur.observation_points.set_op_propagation_speed(uv_op)

                # Store turbine state applied in controller
                c_tmp = self.controller.get_applied_settings(tur, turb_idx, t)
                control_applied = pd.concat([control_applied, c_tmp], ignore_index=True)

                # Store flow field points
                if (self.settings_vis["debug"]["effective_wf_tile"] and
                        t in self.settings_vis["debug"]["time"]):
                    self.visualizer_ff.vis_store_u_values(
                        self.wake_solver.get_tile_u().flatten(), turb_idx)
                    

            lg.info('Rotor wind speed of all turbines:')
            lg.info(uv_r)

            lg.info('Power generated by all turbines: %s' % pow_t)

            # ///////////////////// CORRECT ///////////////////////
            # Load new values for the flow field
            self.ambient_corrector.update(t)
            for turb_idx, tur in enumerate(self.wind_farm.turbines):
                # Apply new values to the turbine states
                self.ambient_corrector(turb_idx, tur.ambient_states)

            # ///////////////////// VISUALIZE /////////////////////
            if (self.settings_vis["debug"]["turbine_effective_wind_speed"] and
                    t in self.settings_vis["debug"]["time"]):
                self.wake_solver.vis_turbine_eff_wind_speed_field(self.wind_farm, self.sim_dir, t)

            # ///////////////////// PROPAGATE /////////////////////
            for turb_idx, tur in enumerate(self.wind_farm.turbines):
                tur.ambient_states.iterate_states_and_keep()
                tur.turbine_states.iterate_states_and_keep()
                tur.observation_points.propagate_ops(self.settings_sim['time step'])
                lg.debug(tur.observation_points.get_world_coord())

            # ///////////////////// CONTROL ///////////////////////
            self.controller.update(t)
            for turb_idx, tur in enumerate(self.wind_farm.turbines):
                lg.debug("Turbine %s states before control-> yaw = %s deg, ax ind = %s." %
                         (turb_idx, tur.turbine_states.get_current_yaw(), tur.turbine_states.get_current_ax_ind()))
                self.controller(tur, turb_idx, t)
                lg.debug("Turbine %s states after control-> yaw = %s deg, ax ind = %s." %
                         (turb_idx, tur.turbine_states.get_current_yaw(), tur.turbine_states.get_current_ax_ind()))
            
            # ====== BART ======
            # /////////////////////// POWER CONTROL ///////////////////////
            if self.power_controller is not None:
                for turb_idx, tur in enumerate(self.wind_farm.turbines):
                    self.power_controller(tur, turb_idx, t, util.ot_uv2abs(uv_r[turb_idx, 0], uv_r[turb_idx, 1]))
            # ====== BART ======

            # ///////////////////// STORE ///////////////////////
            if (self.settings_vis["debug"]["effective_wf_tile"] and
                        t in self.settings_vis["debug"]["time"]):
                self.visualizer_ff.vis_save_flow_field(self.sim_dir + '/flow_field_' + str(t))

            lg.info('Ending time step: %s s.' % t)
            iteration += 1
            self._print_progress_bar(iteration, self.iterations_total, prefix = 'Simulation progress:', suffix = 'Complete', length = 50)

        lg.info('Simulation finished. Resulting measurements:')
        lg.info(measurements)
        return measurements, control_applied

    def set_wind_farm(self, new_wf: wfm.WindFarm):
        """
        Overwrite wind farm object with a new wind farm object. Can be used to restart the simulation from a given state

        Parameters
        ----------
        new_wf :  windfarm.WindFarm
            Wind farm object with turbines and states
        -------
        """
        self.wind_farm = new_wf

    def get_wind_farm(self) -> wfm.WindFarm:
        """
        Get the current wind farm object which equals the simulation state

        Returns
        -------
        windfarm.WindFarm :
            Wind farm object with turbines and states
        """
        return self.wind_farm

    # Print iterations progress
    def _print_progress_bar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()