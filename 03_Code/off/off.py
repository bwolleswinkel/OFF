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
                 settings_cor: dict, settings_ctr: dict, settings_turbine: dict, vis: dict):
        self.wind_farm = wind_farm
        self.settings_sim = settings_sim
        self.settings_vis = vis
        # ====== BART ======
        self.settings_cor = settings_cor
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

        blade_root_bending_moment_t = np.zeros(len(self.wind_farm.turbines))
        edgewise_bending_moment_t = np.zeros(len(self.wind_farm.turbines))
        root_normal_force_t = np.zeros(len(self.wind_farm.turbines))

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

                # Calculate the power generated
                pow_t[turb_idx, :] = tur.calc_power(util.ot_uv2abs(uv_r[turb_idx, 0], uv_r[turb_idx, 1]))
                m_tmp['power_OFF'] = pow_t[turb_idx, :]

                # ====== BART ======
                #: Calculate the loads on the turbine
                try:
                    blade_root_bending_moment_t[turb_idx], edgewise_bending_moment_t[turb_idx], root_normal_force_t[turb_idx] = tur.calc_loads()
                except (AttributeError, NotImplementedError):
                    pass
                # ====== BART ======

                # Add turbine index & timestamp to data
                m_tmp.t_idx = turb_idx
                m_tmp['time'] = t

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

            # ====== BART ======
            if ('rotor_plane_wind_speed' in self.settings_vis["debug"]) and (self.settings_vis["debug"]["rotor_plane_wind_speed"] and np.any([np.allclose(self.settings_vis["debug"]["time"][idx] - t, 0) for idx in range(len(self.settings_vis["debug"]["time"]))])):
                # FIXME: These are mostly proof of concepts
                # TEMP
                #
                print(f"Plotting rotor plane wind speed at t = {t}")
                #
                for turb_idx, tur in enumerate(self.wind_farm.turbines):
                    turb_center_pos = self.wind_farm.get_layout()[turb_idx, :3]
                    tur_yaw_angle = tur.get_yaw_orientation()
                    # TEMP
                    #
                    print(f"Turbine {turb_idx} position: {turb_center_pos}")
                    print(f"Turbine yaw angle: {tur_yaw_angle}")
                    print(f"Turbine rotor diameter: {tur.diameter}")
                    print(f"Effective wind speed at center: {self.wake_solver.floris_wake.vis_tile(np.atleast_1d(turb_center_pos[0]), np.atleast_1d(turb_center_pos[1]), np.atleast_1d(turb_center_pos[2]))}")
                    #
                    # Create a meshgrid without rotation
                    N_sample_points_width = 100
                    N_sample_points_height = 150
                    x = np.linspace(turb_center_pos[0] - tur.diameter/2, turb_center_pos[0] + tur.diameter/2, N_sample_points_width)
                    y = np.array(turb_center_pos[1])
                    z = np.linspace(turb_center_pos[2] - tur.diameter/2, turb_center_pos[2] + tur.diameter/2, N_sample_points_height)
                    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
                    # FIXME: We do some 'transposition' to make it easier to interpret these finding 
                    X, Y, Z = X[0, :, :], Y[0, :, :], Z[0, :, :]
                    X, Y, Z = X.T, Y.T, Z.T
                    X, Y, Z = np.flipud(X), np.flipud(Y), np.flipud(Z)
                    # Now we ROTATE all the values in the array
                    tur_yaw_angle = tur_yaw_angle * ((2 * np.pi) / 360)
                    Rot_around_z_axis = np.array([[np.cos(tur_yaw_angle), np.sin(tur_yaw_angle), 0],
                                                  [-np.sin(tur_yaw_angle), np.cos(tur_yaw_angle), 0],
                                                  [0, 0, 1]])
                    for flow_idx, _ in np.ndenumerate(X):
                        pos_vec = np.array([X[flow_idx], Y[flow_idx], Z[flow_idx]])
                        pos_vec_rot = Rot_around_z_axis @ (pos_vec - turb_center_pos) + turb_center_pos
                        X[flow_idx], Y[flow_idx], Z[flow_idx] = pos_vec_rot[0], pos_vec_rot[1], pos_vec_rot[2]
                    # Pass them to the flow visualizer
                    vels = self.wake_solver.floris_wake.vis_tile(X.flatten(order='F'), Y.flatten(order='F'), Z.flatten(order='F'))
                    vels = np.reshape(vels, (N_sample_points_height, N_sample_points_width), order='F')
                    # TEMP: Plot this flow field
                    #
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    im = ax.imshow(vels, cmap='inferno')
                    plt.colorbar(im, ax=ax)
                    plt.title(f"Rotor Plane Wind Speed at t = {t}, turbine idx = {turb_idx}")
                    plt.xlabel("X Position")
                    plt.ylabel("Z Position")
                    plt.show()
                    #

                    #: Get the azeimuth angle of the turbine
                    try:
                        # FIXME: This is still not working, gives very weird speed estimates, but only at some angle values, and seems to be correct compared to other ones.
                        azimuth_angle = tur.azimuth
                        # TEMP
                        #
                        print(f"Azimuth angle: {np.rad2deg(azimuth_angle)} deg at time {t}")
                        print(f"Rotor speed: {np.rad2deg(tur.omega)} deg/s rpm at time {t}")
                        #
                        #: Extract the velocity profiles along the three blades
                        N_sample_points = 100
                        r = np.linspace(0, tur.diameter / 2, N_sample_points)
                        #: Extract the coordinates to a local coordinate frame
                        x_blade_1, y_blade_1, z_blade_1 = np.full(r.shape, turb_center_pos[0]), \
                                turb_center_pos[1] + r * np.sin(azimuth_angle), \
                                turb_center_pos[2] + r * np.cos(azimuth_angle)
                        x_blade_2, y_blade_2, z_blade_2 = np.full(r.shape, turb_center_pos[0]), \
                                turb_center_pos[1] + r * np.sin(azimuth_angle + np.deg2rad(120)), \
                                turb_center_pos[2] + r * np.cos(azimuth_angle + np.deg2rad(120))
                        x_blade_3, y_blade_3, z_blade_3 = np.full(r.shape, turb_center_pos[0]), \
                                turb_center_pos[1] + r * np.sin(azimuth_angle + np.deg2rad(240)), \
                                turb_center_pos[2] + r * np.cos(azimuth_angle + np.deg2rad(240))
                        #: Rotate these coordinates to the global frame
                        tur_yaw_angle = np.deg2rad(tur_yaw_angle)
                        Rot_around_z_axis = np.array([[np.cos(tur_yaw_angle), np.sin(tur_yaw_angle), 0],
                                                    [-np.sin(tur_yaw_angle), np.cos(tur_yaw_angle), 0],
                                                    [0, 0, 1]])
                        for flow_idx, _ in np.ndenumerate(x_blade_1):
                            pos_vec = np.array([x_blade_1[flow_idx], y_blade_1[flow_idx], z_blade_1[flow_idx]])
                            pos_vec_rot = Rot_around_z_axis @ (pos_vec - turb_center_pos) + turb_center_pos
                            x_blade_1[flow_idx], y_blade_1[flow_idx], z_blade_1[flow_idx] = pos_vec_rot[0], pos_vec_rot[1], pos_vec_rot[2]
                        for flow_idx, _ in np.ndenumerate(x_blade_2):
                            pos_vec = np.array([x_blade_2[flow_idx], y_blade_2[flow_idx], z_blade_2[flow_idx]])
                            pos_vec_rot = Rot_around_z_axis @ (pos_vec - turb_center_pos) + turb_center_pos
                            x_blade_2[flow_idx], y_blade_2[flow_idx], z_blade_2[flow_idx] = pos_vec_rot[0], pos_vec_rot[1], pos_vec_rot[2]
                        for flow_idx, _ in np.ndenumerate(x_blade_3):
                            pos_vec = np.array([x_blade_3[flow_idx], y_blade_3[flow_idx], z_blade_3[flow_idx]])
                            pos_vec_rot = Rot_around_z_axis @ (pos_vec - turb_center_pos) + turb_center_pos
                            x_blade_3[flow_idx], y_blade_3[flow_idx], z_blade_3[flow_idx] = pos_vec_rot[0], pos_vec_rot[1], pos_vec_rot[2]
                        # Get the wind speeds at these locations
                        vel_1 = self.wake_solver.floris_wake.vis_tile(x_blade_1, y_blade_1, z_blade_1)
                        vel_2 = self.wake_solver.floris_wake.vis_tile(x_blade_2, y_blade_2, z_blade_2)
                        vel_3 = self.wake_solver.floris_wake.vis_tile(x_blade_3, y_blade_3, z_blade_3)
                        # TEMP: Plot these profiles
                        #
                        import matplotlib.pyplot as plt
                        _, ax_1 = plt.subplots()
                        ax_1.plot(r, vel_1.flatten(), label=f"Blade 1 (angle = {np.rad2deg(azimuth_angle)} deg)")
                        ax_1.plot(r, vel_2.flatten(), label=f"Blade 2 (angle = {np.rad2deg(azimuth_angle) + 120} deg)")
                        ax_1.plot(r, vel_3.flatten(), label=f"Blade 3 (angle = {np.rad2deg(azimuth_angle) + 240} deg)")
                        ax_1.set_title(f"Rotor Blade Wind Speed at t = {t}, turbine idx = {turb_idx}")
                        ax_1.set_xlabel("Radius (m)")
                        ax_1.set_ylabel("Wind Speed (m/s)")
                        ax_1.legend()
                        #

                        # Convert the wind speeds to loads
                        loads_1 = 0.5 * self.settings_cor['ambient']['air_density'] * (vel_1 ** 2) * tur.drag_coeff * tur.blade_chord(r) * tur.blade_width
                        loads_2 = 0.5 * self.settings_cor['ambient']['air_density'] * (vel_2 ** 2) * tur.drag_coeff * tur.blade_chord(r) * tur.blade_width
                        loads_3 = 0.5 * self.settings_cor['ambient']['air_density'] * (vel_3 ** 2) * tur.drag_coeff * tur.blade_chord(r) * tur.blade_width
                        # Convert the distributed loads to forces and moments on the rotor
                        force_1, force_2, force_3 = np.trapz(loads_1, r), np.trapz(loads_2, r), np.trapz(loads_3, r)
                        moment_1, moment_2, moment_3 = np.trapz(loads_1 * r, r), np.trapz(loads_2 * r, r), np.trapz(loads_3 * r, r)
                        # TEMP
                        #
                        print(f"Resulting forces on the blades: {force_1} N, {force_2} N, {force_3} N")
                        print(f"Resulting moments on the blades: {moment_1} Nm, {moment_2} Nm, {moment_3} Nm")
                        #
                        # TEMP: Plot these load distributions
                        #
                        import matplotlib.pyplot as plt
                        _, ax_2 = plt.subplots()
                        ax_2.plot(r, loads_1.flatten(), label=f"Blade 1 (angle = {np.rad2deg(azimuth_angle)} deg)")
                        ax_2.plot(r, loads_2.flatten(), label=f"Blade 2 (angle = {np.rad2deg(azimuth_angle) + 120} deg)")
                        ax_2.plot(r, loads_3.flatten(), label=f"Blade 3 (angle = {np.rad2deg(azimuth_angle) + 240} deg)")
                        ax_2.set_title(f"Rotor Blade Load Distribution at t = {t}, turbine idx = {turb_idx}")
                        ax_2.set_xlabel("Radius (m)")
                        ax_2.set_ylabel("Load (N)")
                        ax_2.legend()
                        #
                        plt.show()
                    except AttributeError:
                        print("Turbine has no azimuth angle, cannot extract blade specific wind speeds and loads.")
                        pass
                    #

            # ====== BART ======

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