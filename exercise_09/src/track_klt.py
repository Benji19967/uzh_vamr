import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from get_sim_warp import getSimWarp
from get_warped_patch import getWarpedPatch
from scipy.signal import convolve2d


def trackKLT(I_R, I, x_T, r_T, n_iter):
    """
    Input:
        I_R     np.ndarray reference image
        I       np.ndarray image to track points in
        x_T     1 x 2, point to track as [x y] = [col row]
        r_T     scalar, radius of patch to track
        n_iter  scalar, number of iterations
    Output:
        estimated warp
        history of parameter estimates ( 6 x (n_iter + 1) )
            including the initial identity estimate
    """

    pass
    # TODO: Your code here
