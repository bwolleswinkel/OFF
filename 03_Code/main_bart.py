# //////////////////////////////////////////////////////////////////// #
#    ____  ______ ______ 
#   / __ \|  ____|  ____|
#  | |  | | |__  | |__   
#  | |  | |  __| |  __|  
#  | |__| | |    | |     
#   \____/|_|    |_|     
# //////////////////////////////////////////////////////////////////// #

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

# //////////////////////////////////////////////////////////////////// #
# Welcome to the example OFF main file. This showcases how to run a simulation using the OFF framework.
# The settings are defined in the run_example.yaml file, have a look to see what is possible.
# If you experience issues, create a new issue on the GitHub page https://github.com/TUDelft-DataDrivenControl/OFF
# //////////////////////////////////////////////////////////////////// #

import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time

def main():
    start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/run_example_bart.yaml')
    
    # One case used for the publication "A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies" Becker, Lejeune et al. 2024
    #oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/nawea_grid_zp_ki0-02_th5_LuT.yaml')
    
    # Run the simulation
    oi.run_sim()

    print("---OFF Simulation took %s seconds ---" % (time.time() - start_time))

    # Store output
    oi.store_measurements()
    oi.store_applied_control()
    oi.store_run_file()

    # ====== BART ======

    from pathlib import Path
    import yaml

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract the path name
    path_name_measurements = Path(oi.off_sim.sim_dir).name
    path_name_input = Path(f'02_Examples_and_Cases/02_Example_Cases/run_example_bart.yaml')

    # Read the wind direction as two lists
    # FROM: https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    with open(path_name_input) as stream:
        try:
            input_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    wd_input_file = [input_file['ambient']['flow_field']['wind_directions_t'], input_file['ambient']['flow_field']['wind_directions']]

    # Add the power measurements as time-series
    measurements = pd.read_csv(f'runs/{path_name_measurements}/measurements.csv')

    # Extract the power
    power = [measurements.loc[measurements['t_idx'] == idx, 'power_OFF'] for idx in range(9)]

    # ------ PLOTTING ------

    # Plot the power
    fix_power, ax_power = plt.subplots()
    for idx in range(9):
        ax_power.plot(power[idx], label=f'Power {idx:02d}')
    ax_power.set_xlabel('Time [s]')
    ax_power.set_ylabel('Power [W]')
    ax_power.set_title('Power of turbines')
    ax_power.legend()

    # Show the plots
    plt.show()


if __name__ == "__main__":
    main()

