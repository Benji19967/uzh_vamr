import glob

import cv2 as cv
import numpy as np


def save_calibration_matrix(mtx: np.ndarray):
    np.savetxt("data/K.txt", mtx, fmt="%.6f")


def save_poses(rvecs: np.ndarray, tvecs: np.ndarray):
    poses = np.zeros((len(rvecs), 6))
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        poses[i:] = np.array(np.c_[rvec.T, tvec.T])
    np.savetxt("data/poses.txt", poses, fmt="%.12f")

    return poses


def calibrate_camera():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 24

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = sorted(glob.glob("data/iPhone7/*.JPG"))

    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

        print(ret)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (8, 6), corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    save_calibration_matrix(mtx=mtx)
    poses = save_poses(rvecs=rvecs, tvecs=tvecs)

    return mtx, poses


if __name__ == "__main__":
    calibrate_camera()
