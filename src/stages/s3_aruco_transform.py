"""
s3_aruco_transform.py
FrameSequence → CameraWorldTransform

Detects ArUco markers placed flat on the floor and computes a rigid
camera-to-world transform (T_cw). The marker coordinate frame becomes the
world frame, so z=0 is the floor plane by construction.

Base case: static camera. The transform is computed once from the first frame
in which any marker is visible, then locked for the entire sequence.
Multiple markers are supported — all visible markers in the anchor frame are
averaged to give a more stable estimate.
"""
import cv2
import numpy as np

from src.types import FrameSequence, CameraWorldTransform

# 3D corners of a single ArUco marker centred at its own origin (z=0 = floor)
def _marker_object_points(marker_length: float) -> np.ndarray:
    """Return the 4 corner points of a marker in its local frame."""
    h = marker_length / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


def _rvec_tvec_to_T_cw(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rvec/tvec (marker in camera frame) to T_cw (4x4)."""
    R, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4, dtype=np.float32)
    T_cw[:3, :3] = R.T
    T_cw[:3,  3] = (-R.T @ tvec.flatten()).astype(np.float32)
    return T_cw


def estimate_camera_world_transform(
    frames: FrameSequence,
    marker_length: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> CameraWorldTransform:
    """
    Compute and lock a camera-to-world transform from ArUco markers.

    Iterates frames until at least one ArUco marker is detected. If multiple
    markers are visible in the anchor frame, their individual T_cw estimates
    are averaged (translation mean, rotation mean via SVD) for a more stable
    result. The resulting transform is locked for the full sequence.

    Parameters
    ----------
    frames        : FrameSequence from Stage 1
    marker_length : physical side length of the printed marker in metres
    camera_matrix : (3, 3) float64 intrinsics from calibration
    dist_coeffs   : (5,)   float64 distortion coefficients from calibration

    Returns
    -------
    CameraWorldTransform
        T_cw             : (4, 4) float32 locked camera-to-world transform
        locked_frame_idx : index of the frame used to compute the transform
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

    obj_pts = _marker_object_points(marker_length)

    for frame_idx, frame_rgb in enumerate(frames.frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        # Solve PnP for each detected marker
        rotations    = []
        translations = []

        for corner in corners:
            img_pts = corner.reshape(4, 2).astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs
            )
            if not success:
                continue
            R, _ = cv2.Rodrigues(rvec)
            rotations.append(R)
            translations.append(tvec.flatten())

        if not rotations:
            continue

        # Average translations
        mean_t = np.mean(translations, axis=0)

        # Average rotations via SVD on the sum of rotation matrices
        R_sum = np.sum(rotations, axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        mean_R = U @ Vt

        # Build T_cw from averaged R and t
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = (mean_R.T).astype(np.float32)
        T_cw[:3,  3] = (-mean_R.T @ mean_t).astype(np.float32)

        print(f"[ArUco] Locked transform from frame {frame_idx} "
              f"({len(rotations)} marker(s) detected)")

        return CameraWorldTransform(T_cw=T_cw, locked_frame_idx=frame_idx)

    raise RuntimeError(
        "No ArUco markers detected in any frame. "
        "Check that markers are visible and camera calibration is correct."
    )

