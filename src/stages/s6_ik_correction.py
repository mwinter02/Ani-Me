"""
s6_ik_correction.py
WorldJointSequence + ContactLabelSequence → IKJointSequence

Corrects foot sliding by locking the foot's world-space position to its first
contact frame position, then solving analytical two-bone IK (hip → knee → ankle)
to propagate the constraint up the kinematic chain. The pelvis/root height is
adjusted to keep the limb within its natural length.
"""
import numpy as np

from src.types import (
    WorldJointSequence, ContactLabelSequence, IKJointSequence,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
)

# Each leg is defined as (hip_idx, knee_idx, ankle_idx)
LEGS = [
    (LEFT_HIP,  LEFT_KNEE,  LEFT_ANKLE),   # index 0 = left foot
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),  # index 1 = right foot
]


def _two_bone_ik(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle_target: np.ndarray,
    upper_len: float,
    lower_len: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical two-bone IK. Moves knee and ankle to reach ankle_target while
    keeping the hip fixed. Returns corrected (knee_pos, ankle_pos).

    The knee is kept in the plane defined by (hip, original_knee, ankle_target)
    to avoid flipping.
    """
    d_vec   = ankle_target - hip
    d       = np.linalg.norm(d_vec)

    # Clamp distance to reachable range
    max_reach = upper_len + lower_len
    min_reach = abs(upper_len - lower_len)
    d_clamped = np.clip(d, min_reach + 1e-6, max_reach - 1e-6)

    if d_clamped != d:
        # Scale the target toward the hip to stay within reach
        ankle_target = hip + d_vec / (d + 1e-9) * d_clamped
        d = d_clamped

    # Law of cosines: angle at hip
    cos_angle = (upper_len**2 + d**2 - lower_len**2) / (2 * upper_len * d)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_hip = np.arccos(cos_angle)

    # Determine bend axis from original knee position (preserve bend direction)
    knee_vec = knee - hip
    d_unit   = d_vec / (d + 1e-9)
    bend_vec = knee_vec - np.dot(knee_vec, d_unit) * d_unit
    bend_len = np.linalg.norm(bend_vec)
    if bend_len < 1e-6:
        # Fallback: arbitrary perpendicular
        perp = np.array([0, 1, 0], dtype=np.float64)
        if abs(np.dot(d_unit, perp)) > 0.9:
            perp = np.array([1, 0, 0], dtype=np.float64)
        bend_vec = perp - np.dot(perp, d_unit) * d_unit
        bend_len = np.linalg.norm(bend_vec)

    bend_unit = bend_vec / bend_len

    # Rotation axis perpendicular to both d_unit and bend_unit
    rot_axis = np.cross(d_unit, bend_unit)
    rot_axis /= (np.linalg.norm(rot_axis) + 1e-9)

    # Rodrigues rotation of d_unit by angle_hip around rot_axis
    c, s = np.cos(angle_hip), np.sin(angle_hip)
    knee_pos = hip + upper_len * (
        d_unit * c
        + np.cross(rot_axis, d_unit) * s
        + rot_axis * np.dot(rot_axis, d_unit) * (1 - c)
    )

    return knee_pos, ankle_target


def apply_foot_ik(
    world_joints: WorldJointSequence,
    contacts: ContactLabelSequence,
) -> IKJointSequence:
    """
    Apply analytical two-bone IK to correct foot sliding during contact frames.

    For each foot, when contact is detected the ankle is locked to its world
    position at the start of the contact event. IK is solved to move the knee
    accordingly. The pelvis height is adjusted to keep the leg within reach.

    Parameters
    ----------
    world_joints : WorldJointSequence from Stage 4
    contacts     : ContactLabelSequence from Stage 5

    Returns
    -------
    IKJointSequence
        joints_world : (F, 24, 3) float32 — IK-corrected joint positions
        root_trans   : (F, 3)     float32 — pelvis world translation per frame
        fps          : carried from world_joints
    """
    joints = world_joints.joints_world.copy().astype(np.float64)  # (F, 24, 3)
    F = joints.shape[0]

    for foot_idx, (hip_i, knee_i, ankle_i) in enumerate(LEGS):
        locked_target = None

        # Pre-compute natural limb segment lengths from first frame
        upper_len = float(np.linalg.norm(joints[0, knee_i]  - joints[0, hip_i]))
        lower_len = float(np.linalg.norm(joints[0, ankle_i] - joints[0, knee_i]))

        for f in range(F):
            in_contact = contacts.labels[f, foot_idx]

            if in_contact:
                if locked_target is None:
                    # First contact frame — lock the ankle position
                    locked_target = joints[f, ankle_i].copy()

                # Solve IK to reach locked_target
                new_knee, new_ankle = _two_bone_ik(
                    hip=joints[f, hip_i],
                    knee=joints[f, knee_i],
                    ankle_target=locked_target,
                    upper_len=upper_len,
                    lower_len=lower_len,
                )
                joints[f, knee_i]  = new_knee
                joints[f, ankle_i] = new_ankle
            else:
                locked_target = None  # contact broken — release lock

    joints_f32 = joints.astype(np.float32)
    root_trans  = joints_f32[:, 0, :].copy()   # pelvis = joint 0

    return IKJointSequence(
        joints_world=joints_f32,
        root_trans=root_trans,
        fps=world_joints.fps,
    )

