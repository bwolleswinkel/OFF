"""Script to test the evaluation of damage equivalent loads"""

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
import rainflow

# FROM: https://www.youtube.com/watch?v=kOjahdZHL5g  # nopep8


def hysteresis_filtering(a: NDArray, gate_size: float, centering: Literal['mid', 'low', 'high'] = 'mid', remove_nan: bool = True) -> NDArray:
    a = a.copy()
    match centering:
        case 'mid':
            lb, ub = -gate_size / 2, gate_size / 2
        case 'low':
            lb, ub = 0, gate_size
        case 'high':
            lb, ub = -gate_size, 0
    idx_track, idx_test = 0, 1
    while idx_test < a.size:
        diff = a[idx_test] - a[idx_track]
        if lb <= diff and diff <= ub:
            a[idx_test] = np.nan
        else:
            idx_track = idx_test
        idx_test += 1
    if remove_nan:
        a = a[~np.isnan(a)]
    return a


def peak_valley_filtering(a: NDArray, remove_nan: bool = True) -> NDArray:
    """Filters any points that are not peaks or valleys"""
    a = a.copy()
    idx_track, idx_test = 0, 1
    while idx_test < a.size - 1:
        prev_diff = a[idx_test] - a[idx_track]
        next_diff = a[idx_test + 1] - a[idx_test]
        if (prev_diff > 0 and next_diff > 0) or (prev_diff < 0 and next_diff < 0):
            a[idx_test] = np.nan
        else:
            idx_track = idx_test
        idx_test += 1
    if remove_nan:
        a = a[~np.isnan(a)]
    return a


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/29
def discretization(a: NDArray, bin_size: Optional[float] = None, num_bins: Optional[int] = None, datum: Literal['first', 'avg', 'high'] | float = 'first') -> tuple[NDArray, NDArray]:
    """Bin the data in a with the specified bin size or number of bins"""
    if bin_size is not None and num_bins is not None:
        raise ValueError("Only one of 'bin_size' and 'num_bins' can be specified")
    if bin_size is None and num_bins is None:
        num_bins = 10
    valid_data = a[~np.isnan(a)]
    if len(valid_data) == 0:
        raise ValueError("Input array contains only NaN values")
    data_min, data_max = np.min(valid_data), np.max(valid_data)
    if bin_size is None:
        bin_size = (data_max - data_min) / num_bins
    else:
        num_bins = int(np.ceil((data_max - data_min) / bin_size))
    if isinstance(datum, str):
        match datum:
            case 'first':
                start_value = data_min
            case 'avg':
                start_value = (data_min + data_max) / 2 - (num_bins * bin_size) / 2
            case 'high':
                start_value = data_max - num_bins * bin_size
            case _:
                raise ValueError(f"Invalid datum string: {datum}")
    else:
        start_value = float(datum)
    bin_edges = np.array([start_value + i * bin_size for i in range(num_bins)])
    bin_edges_with_upper = np.append(bin_edges, start_value + num_bins * bin_size)
    bin_indices = np.full(a.shape, -1, dtype=int)
    for i, value in enumerate(a):
        if not np.isnan(value):
            bin_idx = np.searchsorted(bin_edges_with_upper[1:], value, side='right')
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            bin_indices[i] = bin_idx
    return bin_edges, bin_indices


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/30
def four_point_count(bin_indices: NDArray, num_bins: int) -> tuple[NDArray, NDArray, float]:
    """Rainflow counting algorithm to count transitions between bins
    
    Arguments
    ---------
    bin_indices
        Array of bin indices from binning function
    num_bins
        Total number of bins
    
    Returns
    -------
    transition_matrix
        Matrix where entry (i,j) represents number of transitions from bin i to bin j
    residue
        Array of points left over from rainflow algorithm
    mean_stress
        Mean stress (average bin index) of the valid data
    """
    # Initialize transition matrix
    transition_matrix = np.zeros((num_bins, num_bins), dtype=int)
    # Remove invalid indices (NaN values are marked as -1)
    valid_indices = bin_indices[bin_indices >= 0]
    if len(valid_indices) < 4:
        mean_stress = np.mean(valid_indices) if len(valid_indices) > 0 else 0.0
        residue = np.array(valid_indices if len(valid_indices) > 0 else [])
        return transition_matrix, residue, mean_stress
    # Calculate mean stress
    mean_stress = np.mean(valid_indices)
    # Convert to list for easier manipulation during rainflow algorithm
    data = valid_indices.tolist()
    residue = []  # Stack to store incomplete cycles
    i = 0
    while i < len(data):
        residue.append(data[i])
        # Apply rainflow algorithm when we have at least 4 points
        while len(residue) >= 4:
            # Get the last 4 points
            y1, y2, y3, y4 = residue[-4], residue[-3], residue[-2], residue[-1]
            # Standard rainflow four-point method: 
            # Check if range |y3-y2| <= |y4-y1| (inner range <= outer range)
            inner_range = abs(y3 - y2)
            outer_range = abs(y4 - y1)
            if inner_range <= outer_range:
                # Extract cycle between points 2 and 3
                # Count transition from y2 to y3 and back (full cycle)
                transition_matrix[y2, y3] += 1
                transition_matrix[y3, y2] += 1
                # Remove points 2 and 3 from residue
                residue.pop(-2)  # Remove y3
                residue.pop(-2)  # Remove y2 (index shifts after first pop)
            else:
                # Cannot extract cycle, break and continue
                break
        i += 1
    # Process remaining residue for half-cycles
    for j in range(len(residue) - 1):
        from_bin = residue[j]
        to_bin = residue[j + 1]
        # Count as half-cycles (you might want to handle this differently)
        transition_matrix[from_bin, to_bin] += 1
    # Return the full residue
    residue = np.array(residue)
    return transition_matrix, residue, mean_stress


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/03/30
def plot_transition_matrix(transition_matrix: NDArray, bin_edges: NDArray, mean_stress: float, show: bool = True) -> Axes:
    """Plot the transition matrix as a heatmap with mean stress lines"""
    mat = transition_matrix.copy().astype(float)
    mat[transition_matrix == 0] = np.nan  # Set zero counts to NaN for better visualization
    fig, ax = plt.subplots()
    num_bins = len(bin_edges)
    bin_size = bin_edges[1] - bin_edges[0] if bin_edges.size > 1 else 1
    load_min, load_max = bin_edges[0], bin_edges[-1] + bin_size
    pos = ax.imshow(mat, cmap='inferno', origin='upper', aspect='auto', 
                    extent=[load_min, load_max, load_max, load_min], vmin=1)
    fig.colorbar(pos, label='Number of transitions', pad=0.2)
    # Add mean stress lines (both axes now show loads)
    mean_stress_load_value = bin_edges[0] + mean_stress * bin_size
    ax.axhline(y=mean_stress_load_value, color='darkgrey', linestyle='--')
    ax.axvline(x=mean_stress_load_value, color='darkgrey', linestyle='--')
    # Calculate bin centers (where ticks should be placed)
    bin_centers = [bin_edges[i] + bin_size / 2 for i in range(num_bins)]
    # Adaptive tick selection - choose subset of bin centers to display
    max_ticks = 10
    if num_bins <= max_ticks:
        # Show all bins if we have few enough
        selected_positions = bin_centers
        selected_indices = list(range(num_bins))
    else:
        # Select evenly spaced bins
        step = max(1, num_bins // max_ticks)
        selected_indices = list(range(0, num_bins, step))
        # Always include the last bin if not already included
        if selected_indices[-1] != num_bins - 1:
            selected_indices.append(num_bins - 1)
        selected_positions = [bin_centers[i] for i in selected_indices]
    # Primary axis: bin numbers on left y-axis and load values on bottom x-axis
    ax.set_xlim(load_min, load_max)
    ax.set_ylim(load_max, load_min)  # Inverted to match matrix orientation
    ax.yaxis.set_major_locator(FixedLocator(selected_positions))
    ax.set_yticklabels([str(i) for i in selected_indices])
    ax.set_ylabel(r"From bin $i$")
    ax.set_xlabel("Load (in Nm)")
    # Secondary axis: bin numbers on top x-axis
    ax_top = ax.twiny()
    ax_top.set_xlim(load_min, load_max)
    ax_top.xaxis.set_major_locator(FixedLocator(selected_positions))
    ax_top.set_xticklabels([str(i) for i in selected_indices])
    ax_top.set_xlabel(r"To bin $j$")
    # Secondary axis: load values on right y-axis
    ax_right = ax.twinx()
    ax_right.set_ylim(load_max, load_min)  # Inverted to match matrix orientation
    ax_right.set_ylabel("Load (in Nm)")
    if show:
        plt.show()
    return ax


def damage_equivalent_load(transition_matrix: NDArray, bin_edges: NDArray, mean_stress: float, wohler_exponent: float, n_ref: float) -> float:
    """Calculate the damage equivalent load from the transition matrix
    
    The DEL is the amplitude of a 1 Hz sine wave that would cause the same
    fatigue damage as the original signal over the reference number of cycles.
    
    Arguments
    ---------
    transition_matrix
        Matrix where entry (i,j) represents number of transitions from bin i to bin j
    bin_edges
        Edges of the bins used for counting
    mean_stress
        Mean stress (average bin index) of the valid data
    wohler_exponent
        S-N curve exponent for the material (Wöhler exponent)
    n_ref
        Reference number of cycles
    
    Returns
    -------
    DEL
        Damage equivalent load in the same units as the bin edges
    """
    # Calculate bin size and centers
    if len(bin_edges) < 2:
        return 0.0
    bin_size = bin_edges[1] - bin_edges[0]
    num_bins = len(bin_edges)
    # Initialize total damage
    total_damage = 0.0
    # Extract stress ranges and cycle counts from transition matrix
    for i in range(num_bins):
        for j in range(num_bins):
            if i != j and transition_matrix[i, j] > 0:
                # Convert bin indices to stress values (use bin centers)
                stress_i = bin_edges[i] + bin_size / 2
                stress_j = bin_edges[j] + bin_size / 2
                # Calculate stress range (amplitude)
                stress_range = abs(stress_j - stress_i)
                # Number of cycles at this stress range
                # NOTE: rainflow algorithm counts transitions, so we have half-cycles
                n_cycles = transition_matrix[i, j] / 2.0  # Convert transitions to cycles
                # Calculate damage contribution using S-N curve
                # Damage = n / N, where N ∝ 1/S^m
                # For DEL calculation, we use: damage ∝ n * S^m
                if stress_range > 0:  # Avoid division by zero
                    damage_contribution = n_cycles * (stress_range ** wohler_exponent)
                    total_damage += damage_contribution
    # Calculate damage equivalent load
    # DEL^m * n_ref = total_damage
    # DEL = (total_damage / n_ref)^(1/m)
    if total_damage > 0 and n_ref > 0:
        del_value = (total_damage / n_ref) ** (1.0 / wohler_exponent)
    else:
        del_value = 0.0
    return del_value
        

def main():
    # Set the parameters
    n = 1_000
    gate_size = 0
    bin_size = 0.1
    wohler_exponent = 10

    # Create the synthetic load signal
    t = np.linspace(0, 10, n)
    data = np.sin(2 * np.pi * t) + 0.1 * np.random.normal(size=n) + np.sin(2 * np.pi * 0.5 * t + 2 * np.random.rand(n))

    # Apply hysteresis filtering
    data_hys = hysteresis_filtering(data, gate_size=gate_size, remove_nan=False)
    t_hys = t[~np.isnan(data_hys)]
    data_hys = data_hys[~np.isnan(data_hys)]

    # Apply peak-valley filtering
    data_peak_valley = peak_valley_filtering(data_hys, remove_nan=False)
    t_peak_valley = t_hys[~np.isnan(data_peak_valley)]
    data_peak_valley = data_peak_valley[~np.isnan(data_peak_valley)]

    # Apply discretization
    bin_edges, bin_indices = discretization(data_peak_valley, bin_size=bin_size)

    # Apply rainflow counting algorithm
    transition_matrix, residue, mean_stress = four_point_count(bin_indices, num_bins=bin_edges.size)

    # Calculate damage equivalent load
    signal_duration = t[-1] - t[0]  # Duration of signal in seconds
    reference_frequency = 1.0  # 1 Hz reference frequency
    n_ref = reference_frequency * signal_duration  # Reference number of cycles
    
    del_value = damage_equivalent_load(transition_matrix, bin_edges, mean_stress, wohler_exponent, n_ref)

    # Plot the data and the filtered results
    fig, (ax_org, ax_hys, ax_peak_valley) = plt.subplots(3, 1, sharex=True)
    ax_org.plot(t, data, label='Original Data')
    ax_hys.plot(t_hys, data_hys, label='Hysteresis Filtered', color='orange')
    ax_peak_valley.plot(t_peak_valley, data_peak_valley, label='Peak-Valley Filtered', color='green')
    ax_org.legend(loc='upper left')
    ax_hys.legend(loc='upper left')
    ax_peak_valley.legend(loc='upper left')

    # Plot the transition matrix
    plot_transition_matrix(transition_matrix, bin_edges, mean_stress, show=True)

    # Print the damage equivalent load
    print(f"Damage Equivalent Load (DEL): {del_value:.2f} (in the same units as bin edges)")
    

if __name__ == '__main__':
    main()

