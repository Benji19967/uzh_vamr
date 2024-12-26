from pathlib import Path

import cv2
import numpy as np


def read_img(filepath: Path) -> np.ndarray:
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # type: ignore
