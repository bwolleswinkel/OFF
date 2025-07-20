""""This is a file with handy utils

"""

import numpy as np

# ------------ DATA ------------

# Tableau colors
# FROM: https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
tableau_color_palette_10 = ["#4E79A7",  # 0: Blue
                            "#F28E2B",  # 1: Orange
                            "#E15759",  # 2: Red
                            "#76B7B2",  # 3: Teal
                            "#59A14F",  # 4: Green
                            "#EDC948",  # 5: Yellow
                            "#B07AA1",  # 6: Purple
                            "#FF9DA7",  # 7: Pink
                            "#9C755F",  # 8: Brown
                            "#BAB0AC"]  # 9: Gray

# ------------ FUNCTIONS ------------

def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another."""
    match (from_unit, to_unit):
        case ('RPM', 'rad/s'):
            return value * ((2 * np.pi) / 60)
        case ('rad/s', 'RPM'):
            return value * (60 / (2 * np.pi))
        case ('deg', 'rad'):
            return value * ((2 * np.pi) / 360)
        case ('rad', 'deg'):
            return value * (360 / (2 * np.pi))
        case _:
            raise ValueError(f"Unsupported conversion from '{from_unit}' to '{to_unit}'")