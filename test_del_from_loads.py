"""Script to test generating a DEL and accrued damage from a time series of loads"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import rainflow


def compute_del(load: NDArray, m: float, n_eq: float) -> float:
    """Compute DEL from a load signal using rainflow cycle extraction"""
    if m <= 0:
        raise ValueError("Wohler exponent m must be > 0")
    if n_eq <= 0:
        raise ValueError("Equivalent cycle count n_eq must be > 0")

    damage_sum = 0.0
    for stress_range, _, cycle_count, _, _ in rainflow.extract_cycles(load):
        damage_sum += cycle_count * stress_range ** m

    return (damage_sum / n_eq) ** (1 / m)


def equivalent_1hz_load(time_s: NDArray, del_range: float, offset: float, frequency_hz: float = 1.0, phase_rad: float = 0.0) -> tuple[NDArray, float]:
    """Build a 1 Hz equivalent sinusoid from DEL range and chosen offset"""
    # rainflow.extract_cycles reports cycle ranges, so DEL here is also a range.
    amplitude = 0.5 * del_range
    eq_signal = offset + amplitude * np.sin(2 * np.pi * frequency_hz * time_s + phase_rad)
    return eq_signal, amplitude


def damage_from_del(del_value: float, m: float, n_eq: float) -> float:
    """Map DEL back to accumulated damage proxy D = DEL^m * N_eq"""
    return float(del_value ** m * n_eq)


if __name__ == '__main__':
    # Import file
    with open(Path('runs') / 'off_run_D2026_06_03_T17_04_01' / 'measurements.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(8, 17))
        # FIXME: Half the data
        data = data[:data.shape[0] // 5, :]
        time_s, load_knm = data[1::3, 0], data[1::3, 1] / 1E3

    # DEL settings
    wohler_exponent = 10.0  # Polymer default
    reference_frequency_hz = 1.0
    signal_duration_s = time_s[-1] - time_s[0]
    n_eq = reference_frequency_hz * signal_duration_s

    del_knm = compute_del(load_knm, m=wohler_exponent, n_eq=n_eq)
    mean_knm = np.mean(load_knm)
    eq_1hz_knm, eq_amplitude_knm = equivalent_1hz_load(time_s, del_knm, offset=mean_knm, frequency_hz=reference_frequency_hz)
    damage_proxy = damage_from_del(del_knm, m=wohler_exponent, n_eq=n_eq)

    print(f"Samples: {load_knm.size}")
    print(f"Signal duration: {signal_duration_s:.2f} s")
    print(f"N_eq (1 Hz): {n_eq:.2f} cycles")
    print(f"DEL range: {del_knm:.3f} kNm")
    print(f"Equivalent 1 Hz amplitude: {eq_amplitude_knm:.3f} kNm")
    print(f"Equivalent 1 Hz offset: {mean_knm:.3f} kNm")
    print(f"Damage proxy D = DEL^m * N_eq: {damage_proxy:.3e}")

    # Plot the timeseries
    fig, ax = plt.subplots()
    ax.plot(time_s, load_knm, label='Original load')
    ax.plot(time_s, eq_1hz_knm, 'r--', label='Equivalent 1 Hz load')
    ax.set_xlabel("Time (in s)")
    ax.set_ylabel("Blade root bending moment (in kNm)")
    ax.legend()

    # Sanity check
    sgn_test = 1E4 * np.sin(time_s * 2 * np.pi * reference_frequency_hz) + 1E5

    del_test = compute_del(sgn_test, m=wohler_exponent, n_eq=n_eq)
    _, amp_test = equivalent_1hz_load(time_s, del_test, offset=float(np.mean(sgn_test)), frequency_hz=reference_frequency_hz)

    print(f"Test DEL range: {del_test:.2E} kNm (should be close to 2.0E+04)")
    print(f"Test equivalent amplitude: {amp_test:.2E} kNm (should be close to 1.0E+04)")

    plt.show()
