import numpy as np
from fundamental_eight_point import fundamentalEightPoint
from fundamental_eight_point_normalized import fundamentalEightPointNormalized

from utils import distPoint2EpipolarLine

# Number of 3D points to test
N = 40

# Random homogeneous coordinates of 3-D points
#  X = np.random.rand(4,N)
p_W_hom_random = np.loadtxt("matlab_X.csv", delimiter=",")

# Simulated scene with error-free correspondances
p_W_hom_random[2, :] = p_W_hom_random[2, :] * 5 + 10
p_W_hom_random[3, :] = 1

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

# Image (i.e. projected points)
p1_P_hom = M1 @ p_W_hom_random
p2_P_hom = M2 @ p_W_hom_random

sigma = 1e-1
#  p1_P_hom_noisy = p1_P_hom + sigma * np.random.randn(*p1_P_hom.shape)
#  p2_P_hom_noisy = p2_P_hom + sigma * np.random.randn(*p2_P_hom.shape)

# If you want to get the same results as matlab users, uncomment those two lines
p1_P_hom_noisy = np.loadtxt("matlab_noisy_x1.csv", delimiter=",")
p2_P_hom_noisy = np.loadtxt("matlab_noisy_x2.csv", delimiter=",")

# Estimate Fundamental Matrix via 8-point algorithm
F = fundamentalEightPoint(p1_P_hom, p2_P_hom)
cost_algebraic = np.linalg.norm(np.sum(p2_P_hom * (F @ p1_P_hom))) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, p1_P_hom, p2_P_hom)

print("")
print("Noise-free correspondences")
print("Algebraic error: %f" % cost_algebraic)
print("Geometric error: %f px" % cost_dist_epi_line)

# Test with noise
F = fundamentalEightPoint(p1_P_hom_noisy, p2_P_hom_noisy)  # This gives bad results!

cost_algebraic = np.linalg.norm(
    np.sum(p2_P_hom_noisy * (F @ p1_P_hom_noisy), axis=0)
) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, p1_P_hom_noisy, p2_P_hom_noisy)

print("")
print("Noisy correspondences with 8 Point Algorithm")
print("Algebraic error: %f" % cost_algebraic)
print("Geometric error: %f px" % cost_dist_epi_line)

# Test with noise
# F = fundamentalEightPoint(p1_P_hom_noisy, p2_P_hom_noisy)  # This gives bad results!
F = fundamentalEightPointNormalized(
    p1_P_hom_noisy, p2_P_hom_noisy
)  # This gives good results!

cost_algebraic = np.linalg.norm(
    np.sum(p2_P_hom_noisy * (F @ p1_P_hom_noisy), axis=0)
) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, p1_P_hom_noisy, p2_P_hom_noisy)

print("")
print("Noisy correspondences with normalized 8 Point Algorithm")
print("Algebraic error: %f" % cost_algebraic)
print("Geometric error: %f px" % cost_dist_epi_line)
