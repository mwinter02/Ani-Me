"""
s5_contact_labels.py
WorldJointSequence → ContactLabelSequence

Classifies each foot as in-contact or not per frame using two simultaneous
conditions on world-space joint positions:
  1. Height threshold  — foot joint z-height is within ε of the floor (z ≈ 0)
  2. Velocity threshold — foot joint speed between adjacent frames is low

Contact is true if EITHER the ankle OR the toe joint satisfies both conditions.
"""
import numpy as np

from src.types import (
    WorldJointSequence, ContactLabelSequence,
    LEFT_ANKLE, LEFT_TOE, RIGHT_ANKLE, RIGHT_TOE,
)


def detect_foot_contacts(
    world_joints: WorldJointSequence,
    height_thresh: float = 0.05,
    vel_thresh: float = 0.20,
) -> ContactLabelSequence:
    """
    Detect per-frame foot ground contacts.

    Parameters
    ----------
    world_joints   : WorldJointSequence from Stage 4
    height_thresh  : max z-height (metres) to consider a foot grounded
    vel_thresh     : max joint speed (metres/frame) to consider a foot stationary

    Returns
    -------
    ContactLabelSequence
        labels : (F, 2) bool  — columns are [left_foot, right_foot]
        fps    : carried from world_joints
    """
    joints = world_joints.joints_world   # (F, 24, 3)
    F = joints.shape[0]

    # Joint velocity: zero-pad first frame (no previous frame)
    velocity = np.zeros_like(joints)                           # (F, 24, 3)
    velocity[1:] = np.abs(joints[1:] - joints[:-1])            # (F-1, 24, 3)
    speed = np.linalg.norm(velocity, axis=-1)                  # (F, 24)

    def _contact_for_joints(ankle_idx: int, toe_idx: int) -> np.ndarray:
        """Return (F,) bool contact array for one foot."""
        ankle_height  = joints[:, ankle_idx, 2]   # (F,)
        toe_height    = joints[:, toe_idx,   2]   # (F,)
        ankle_speed   = speed[:, ankle_idx]        # (F,)
        toe_speed     = speed[:, toe_idx]           # (F,)

        ankle_contact = (ankle_height < height_thresh) & (ankle_speed < vel_thresh)
        toe_contact   = (toe_height   < height_thresh) & (toe_speed   < vel_thresh)

        return ankle_contact | toe_contact            # (F,) bool

    left_contact  = _contact_for_joints(LEFT_ANKLE,  LEFT_TOE)
    right_contact = _contact_for_joints(RIGHT_ANKLE, RIGHT_TOE)

    labels = np.stack([left_contact, right_contact], axis=1)  # (F, 2)

    return ContactLabelSequence(labels=labels, fps=world_joints.fps)

