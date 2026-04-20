"""
calibration.py
Load OpenCV camera intrinsics from a .npz calibration file.

Produce the calibration file once with:
    cv2.calibrateCamera() using a checkerboard pattern
    np.savez(path, camera_matrix=K, dist_coeffs=d)
"""
import numpy as np


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics from a .npz file.

    Parameters
    ----------
    path : str
        Path to the .npz file containing 'camera_matrix' and 'dist_coeffs'.

    Returns
    -------
    camera_matrix : (3, 3) float64
    dist_coeffs   : (5,)   float64
    """
    data = np.load(path)
    camera_matrix = data["camera_matrix"].astype(np.float64)
    dist_coeffs   = data["dist_coeffs"].astype(np.float64).flatten()
    return camera_matrix, dist_coeffs

