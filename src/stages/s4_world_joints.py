"""
s4_world_joints.py
SMPLSequence + CameraWorldTransform → WorldJointSequence

Applies the locked camera-to-world transform to all per-frame joint positions,
converting them from camera space to world space. z=0 is the floor by
construction (defined by the ArUco marker plane).
"""
import numpy as np

from src.types import SMPLSequence, CameraWorldTransform, WorldJointSequence


def to_world_joints(
    smpl: SMPLSequence,
    transform: CameraWorldTransform,
) -> WorldJointSequence:
    """
    Project joint positions from camera space to world space.

    Homogeneous-extends joints_cam to (F, 24, 4), left-multiplies by T_cw,
    and strips the homogeneous column.

    Parameters
    ----------
    smpl      : SMPLSequence from Stage 2
    transform : CameraWorldTransform from Stage 3

    Returns
    -------
    WorldJointSequence
        joints_world : (F, 24, 3) float32 in world space, z=0 is floor, metres
        fps          : carried from FrameSequence (caller must pass fps through)
    """
    F, J, _ = smpl.joints_cam.shape                       # (F, 24, 3)
    ones     = np.ones((F, J, 1), dtype=np.float32)
    joints_h = np.concatenate([smpl.joints_cam, ones], axis=-1)  # (F, 24, 4)

    # T_cw is (4, 4); apply to each frame: (4,4) @ (4, N)^T
    T = transform.T_cw  # (4, 4)
    joints_world = (T @ joints_h.reshape(-1, 4).T).T  # (F*24, 4)
    joints_world = joints_world[:, :3].reshape(F, J, 3).astype(np.float32)

    # Sanity check: warn if foot joints are consistently below the floor
    from src.types import LEFT_ANKLE, RIGHT_ANKLE, LEFT_TOE, RIGHT_TOE
    foot_indices = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_TOE, RIGHT_TOE]
    min_z = joints_world[:, foot_indices, 2].min()
    if min_z < -0.05:
        print(f"[WorldJoints] WARNING: foot joints penetrate floor by up to "
              f"{abs(min_z):.3f}m — check scale or calibration.")

    # fps is not stored on SMPLSequence; caller provides it via pipeline.py
    # Return with fps=0.0 as a sentinel; pipeline.py fills it in.
    return WorldJointSequence(joints_world=joints_world, fps=0.0)

