"""
There are two (common) ways to represent n points in an m-dimensional using matrices.

Example: we take the 3 points (1, 2), (3, 4), (5, 6) in 2-dimensional space.

Option 1) nxm:
[
    [1, 2],
    [3, 4],
    [5, 6]
]

Option 2) mxn:
[
    [1, 3, 5],
    [2, 4, 6]
]

Which one is better?

Option 2 has its advantages: applying transformation to points
"""

import numpy as np


class Points:
    def __init__(self, p: np.ndarray) -> None:
        self.p = p
