"""Script to test the different standards of YAML parsing tabular structures"""

import yaml
import numpy as np


with open('test_yaml_tabular.yaml', 'r') as f:
    data = yaml.safe_load(f)
    timeseries_two_dashes = []
    for key, value in data.items():
        if key == 'timeseries_two_dashes':
            timeseries_two_dashes.append(value)
        elif key == 'timeseries_two_dashes_t':
            timeseries_two_dashes.append(value)
        else:
            exec(f'{key} = value')
    for key in data.keys():
        var = None
        if key == 'timeseries_two_dashes':
            print(f'\n=== {key} ===')
            print(f"{timeseries_two_dashes} | shape={np.array(timeseries_two_dashes).shape}")
        elif key == 'timeseries_two_dashes_t':
            pass
        else:
            print(f'\n=== {key} ===')
            exec(f'var = {key}')
            if isinstance(var, float):
                print(f"{var} | shape=N/A)")
            else:
                try:
                    print(f"{var} | shape={np.array(var).shape}")
                except ValueError as _:
                    print(f"{var} | shape=N/A)")