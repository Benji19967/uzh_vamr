import numpy as np
from scipy.optimize import least_squares

from utils import HomogMatrix2twist, twist2HomogMatrix


def alignEstimateToGroundTruth(pp_G_C, p_V_C):
    """
    Returns the points of the estimated trajectory p_V_C transformed into the ground truth frame G.
    The similarity transform Sim_G_V is to be chosen such that it results in the lowest error between
    the aligned trajectory points p_G_C and the points of the ground truth trajectory pp_G_C.
    All matrices are 3xN
    """
    # TODO: Your code here
