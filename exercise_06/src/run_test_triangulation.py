import numpy as np
from linear_triangulation import linearTriangulation

# Number of 3D points to test
N = 10

# Random homogeneous coordinates of 3-D points
p_W_hom = np.random.rand(4, N)
p_W_hom[2, :] = p_W_hom[2, :] * 5 + 10
p_W_hom[3, :] = 1

# Test linear triangulation
M1 = np.array(
    [
        [500, 0, 320, 0],
        [0, 500, 240, 0],
        [0, 0, 1, 0],
    ]
)

M2 = np.array(
    [
        [500, 0, 320, -100],
        [0, 500, 240, 0],
        [0, 0, 1, 0],
    ]
)

p1_P_hom = M1 @ p_W_hom
p2_P_hom = M2 @ p_W_hom

p_W_hom_est = linearTriangulation(p1_P_hom, p2_P_hom, M1, M2)

print("p_W_hom_est - p_W_hom")
print(p_W_hom_est - p_W_hom)
print(
    "Your function looks %s"
    % ("Correct" if np.allclose(p_W_hom_est, p_W_hom) else "Incorrect")
)
