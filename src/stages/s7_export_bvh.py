"""
s7_export_bvh.py
IKJointSequence → BVH file

Converts the IK-corrected joint sequence to a BVH animation file.
Joint world positions are converted to local-frame Euler angles (ZXY order)
relative to each joint's parent. The BVH HIERARCHY is built from skeleton_def,
and the MOTION section is written frame by frame.
"""
import numpy as np
from typing import Any

from src.types import IKJointSequence, SMPL_JOINT_NAMES

# Default SMPL → BVH skeleton definition.
# Each entry: (joint_name, parent_index, rest_offset_metres)
# rest_offset is the joint's position relative to its parent in the rest pose.
DEFAULT_SKELETON_DEF: dict[str, Any] = {
    "joint_names":    SMPL_JOINT_NAMES,
    "parent_indices": [
        -1,  # 0  pelvis      (root)
         0,  # 1  left_hip
         0,  # 2  right_hip
         0,  # 3  spine1
         1,  # 4  left_knee
         2,  # 5  right_knee
         3,  # 6  spine2
         4,  # 7  left_ankle
         5,  # 8  right_ankle
         6,  # 9  spine3
         7,  # 10 left_foot
         8,  # 11 right_foot
         9,  # 12 neck
         9,  # 13 left_collar
         9,  # 14 right_collar
        12,  # 15 head
        13,  # 16 left_shoulder
        14,  # 17 right_shoulder
        16,  # 18 left_elbow
        17,  # 19 right_elbow
        18,  # 20 left_wrist
        19,  # 21 right_wrist
        20,  # 22 left_hand
        21,  # 23 right_hand
    ],
}


def _rotation_matrix_to_euler_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to ZXY Euler angles (degrees)."""
    # ZXY convention: R = Rz * Rx * Ry
    x = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    if abs(np.cos(x)) > 1e-6:
        y = np.arctan2(-R[2, 0], R[2, 2])
        z = np.arctan2(-R[0, 1], R[1, 1])
    else:
        y = 0.0
        z = np.arctan2(R[0, 2], R[0, 0])
    return np.degrees(z), np.degrees(x), np.degrees(y)


def export_bvh(
    ik_joints: IKJointSequence,
    output_path: str,
    skeleton_def: dict | None = None,
) -> None:
    """
    Write an IKJointSequence to a BVH animation file.

    Parameters
    ----------
    ik_joints    : IKJointSequence from Stage 6
    output_path  : destination .bvh file path
    skeleton_def : joint hierarchy definition (defaults to SMPL 24-joint layout)
                   Keys: 'joint_names' (list[str]), 'parent_indices' (list[int])
    """
    if skeleton_def is None:
        skeleton_def = DEFAULT_SKELETON_DEF

    joint_names    = skeleton_def["joint_names"]
    parent_indices = skeleton_def["parent_indices"]
    J = len(joint_names)

    joints = ik_joints.joints_world   # (F, 24, 3)
    F      = joints.shape[0]
    fps    = ik_joints.fps
    dt     = 1.0 / fps if fps > 0 else 1.0 / 30.0

    # Compute rest-pose offsets from frame 0
    rest_offsets = []
    for j, parent in enumerate(parent_indices):
        if parent == -1:
            rest_offsets.append(joints[0, j])   # root: absolute position
        else:
            rest_offsets.append(joints[0, j] - joints[0, parent])

    lines = []

    # -----------------------------------------------------------------------
    # HIERARCHY
    # -----------------------------------------------------------------------
    lines.append("HIERARCHY")

    indent_stack: list[int] = []

    def _write_joint(j: int, indent: int) -> None:
        pad = "\t" * indent
        name = joint_names[j]
        is_root = (parent_indices[j] == -1)
        children = [c for c in range(J) if parent_indices[c] == j]

        lines.append(f"{pad}{'ROOT' if is_root else 'JOINT'} {name}")
        lines.append(f"{pad}{{")

        off = rest_offsets[j] * 100  # convert metres → centimetres for BVH
        lines.append(f"{pad}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")

        if is_root:
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Xrotation Yrotation")

        if children:
            for child in children:
                _write_joint(child, indent + 1)
        else:
            # End site
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{pad}\t}}")

        lines.append(f"{pad}}}")

    root_joints = [j for j in range(J) if parent_indices[j] == -1]
    for rj in root_joints:
        _write_joint(rj, 0)

    # -----------------------------------------------------------------------
    # MOTION
    # -----------------------------------------------------------------------
    lines.append("MOTION")
    lines.append(f"Frames: {F}")
    lines.append(f"Frame Time: {dt:.6f}")

    for f in range(F):
        frame_values = []

        for j in range(J):
            parent = parent_indices[j]

            if parent == -1:
                # Root: write translation (cm) + rotation
                pos = joints[f, j] * 100  # metres → centimetres
                frame_values.extend([pos[0], pos[1], pos[2]])
                # Root rotation: identity for now (global orientation from theta)
                frame_values.extend([0.0, 0.0, 0.0])
            else:
                # Compute local rotation from child-parent vector change
                rest_dir = rest_offsets[j]
                rest_len = np.linalg.norm(rest_dir)

                curr_dir = joints[f, j] - joints[f, parent]
                curr_len = np.linalg.norm(curr_dir)

                if rest_len > 1e-6 and curr_len > 1e-6:
                    a = rest_dir / rest_len
                    b = curr_dir / curr_len
                    axis = np.cross(a, b)
                    axis_len = np.linalg.norm(axis)
                    if axis_len > 1e-6:
                        axis /= axis_len
                        angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
                        # Build rotation matrix via Rodrigues
                        c, s = np.cos(angle), np.sin(angle)
                        K = np.array([
                            [0,        -axis[2],  axis[1]],
                            [axis[2],   0,       -axis[0]],
                            [-axis[1],  axis[0],  0],
                        ])
                        R = np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K
                        rz, rx, ry = _rotation_matrix_to_euler_zxy(R)
                        frame_values.extend([rz, rx, ry])
                    else:
                        frame_values.extend([0.0, 0.0, 0.0])
                else:
                    frame_values.extend([0.0, 0.0, 0.0])

        lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Export] BVH written to {output_path}  ({F} frames @ {fps:.1f} fps)")

