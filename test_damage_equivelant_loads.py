"""Script to test the evaluation of damage equivalent loads"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def reduce(a: NDArray, size: float, centering: Literal['mid', 'low', 'high'] = 'mid', remove_nan: bool = True) -> NDArray:
    a = a.copy()
    match centering:
        case 'mid':
            lb, ub = -size / 2, size / 2
        case 'low':
            lb, ub = 0, size
        case 'high':
            lb, ub = -size, 0
    idx_track, idx_test = 0, 1
    while idx_test < a.size:
        diff = a[idx_track] - a[idx_test]
        if lb <= diff and diff <= ub:
            a[idx_test] = np.nan
        else:
            idx_track = idx_test
        idx_test += 1
    if remove_nan:
        a = a[~np.isnan(a)]
    return a


def main():
    # Check the loads on turbine 4, 6, and 7
    a = np.random.rand(100) * np.sin(np.linspace(1, 10, 100))

    b = reduce(a, 0.5, remove_nan=False)

    plt.plot(a)
    plt.plot(b)
    plt.show()


if __name__ == '__main__':
    main()

