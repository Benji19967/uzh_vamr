import numpy as np

NUM_ITERATIONS = 1000
S = 3


def parabolaRansac(data: np.ndarray, max_noise):
    """
    Apply Ransac to get fit for a parabola

    :param data: (2, 30) x and y coordinates of 30 points (20 inliers, 10 outliers)
    :param max_noise:
    :returns: (best_guesses, max_num_inliers)

        - best_guesses has shape (3, NUM_ITERATIONS) with the polynome coefficients
        from polyfit of the best guess so far at each iteration columnwise and

        - max_num_inliers has shape (1, NUM_ITERATIONS), with the inlier count of the
        best guess so far at each iteration.
    """
    num_points = data.shape[1]

    best_guesses = np.zeros((3, NUM_ITERATIONS + 1))
    max_num_inliers = [0] * (NUM_ITERATIONS + 1)
    inliers_of_max_num_inliers = None
    for i in range(NUM_ITERATIONS):
        pts_indexes = np.random.choice(np.arange(num_points), size=S, replace=False)
        pts = data[:, pts_indexes]
        pts_x = pts[0]
        pts_y = pts[1]
        coefficients = np.polyfit(x=pts_x, y=pts_y, deg=2)

        y_vals_fitted = np.polyval(coefficients, data[0])
        inliers = data[:, np.abs(data[1] - y_vals_fitted) <= max_noise]

        num_inlier_points = inliers.shape[1]
        if num_inlier_points > max_num_inliers[i]:
            max_num_inliers[i + 1] = num_inlier_points
            best_guesses[:, i + 1] = coefficients
            inliers_of_max_num_inliers = inliers
        else:
            max_num_inliers[i + 1] = max_num_inliers[i]
            best_guesses[:, i + 1] = best_guesses[:, i]

    pts_x = inliers_of_max_num_inliers[0]
    pts_y = inliers_of_max_num_inliers[1]
    coefficients = np.polyfit(x=pts_x, y=pts_y, deg=2)
    best_guesses[:, -1] = coefficients
    max_num_inliers[-1] = max_num_inliers[-2]

    print(best_guesses, max_num_inliers)
    return best_guesses, max_num_inliers
