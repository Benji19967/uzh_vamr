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

Opinions on the web:
https://www.chrishecker.com/Column_vs_Row_Vectors#:~:text=When%20you're%20doing%20math,a%20single%20simple%20example...
- "When you're doing math for graphics, physics, games, or whatever, you should use column vectors when you're representing points, differences between points, and the like"

https://steve.hollasch.net/cgindex/math/matrix/column-vec.html
- "Recent mathematical treatments of linear algebra and related fields invariably treat vectors as columns"

https://toqoz.fyi/matrix-math-confusion.html
- "Post-multiply (column vectors) is the most dominant by far, especially in mathematical texts and really everything other than computer graphics.  It represents the vector as a single column matrix on the right side"
"""

import numpy as np


class Points:
    def __init__(self, p: np.ndarray) -> None:
        self.p = p
