from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# SMPL joint index constants (24-joint body model)
# ---------------------------------------------------------------------------
LEFT_HIP    = 1
RIGHT_HIP   = 2
LEFT_KNEE   = 4
RIGHT_KNEE  = 5
LEFT_ANKLE  = 7
RIGHT_ANKLE = 8
LEFT_TOE    = 10
RIGHT_TOE   = 11

SMPL_JOINT_NAMES = [
    "pelvis",       # 0
    "left_hip",     # 1
    "right_hip",    # 2
    "spine1",       # 3
    "left_knee",    # 4
    "right_knee",   # 5
    "spine2",       # 6
    "left_ankle",   # 7
    "right_ankle",  # 8
    "spine3",       # 9
    "left_foot",    # 10
    "right_foot",   # 11
    "neck",         # 12
    "left_collar",  # 13
    "right_collar", # 14
    "head",         # 15
    "left_shoulder",# 16
    "right_shoulder",#17
    "left_elbow",   # 18
    "right_elbow",  # 19
    "left_wrist",   # 20
    "right_wrist",  # 21
    "left_hand",    # 22
    "right_hand",   # 23
]


# ---------------------------------------------------------------------------
# Pipeline dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FrameSequence:
    """
    Decoded video frames at a fixed framerate.

    frames      : (F, H, W, 3) uint8 RGB
    fps         : frames per second (fixed — set at extraction time)
    source_path : original video file path
    """
    frames: np.ndarray      # (F, H, W, 3) uint8
    fps: float
    source_path: str


@dataclass
class SMPLSequence:
    """
    Per-frame SMPL parameters in camera space.

    theta      : (F, 72) float32 — pose parameters (24 joints × 3 axis-angle)
    beta       : (10,)   float32 — shape parameters (median across frames)
    joints_cam : (F, 24, 3) float32 — joint positions in camera space, metres
    n_frames   : number of frames F
    """
    theta: np.ndarray       # (F, 72)
    beta: np.ndarray        # (10,)
    joints_cam: np.ndarray  # (F, 24, 3)
    n_frames: int


@dataclass
class CameraWorldTransform:
    """
    Rigid camera-to-world transform derived from ArUco markers.
    The marker plane defines z=0 (floor).

    T_cw             : (4, 4) float32 — camera-to-world homogeneous transform
    locked_frame_idx : index of the frame used to compute and lock the transform
    """
    T_cw: np.ndarray        # (4, 4) float32
    locked_frame_idx: int


@dataclass
class WorldJointSequence:
    """
    Joint positions in world space after applying CameraWorldTransform.
    z=0 is the floor plane. Units are metres.

    joints_world : (F, 24, 3) float32 — world-space joint positions
    fps          : frames per second (carried from FrameSequence)
    """
    joints_world: np.ndarray  # (F, 24, 3)
    fps: float


@dataclass
class ContactLabelSequence:
    """
    Per-frame binary foot contact labels.

    labels : (F, 2) bool — axis-1 is [left_foot, right_foot]
             True = foot is in contact with the ground
    fps    : frames per second (carried through for velocity reference)
    """
    labels: np.ndarray  # (F, 2) bool
    fps: float


@dataclass
class IKJointSequence:
    """
    IK-corrected joint positions in world space.
    Root translation is separated out for clean BVH export.

    joints_world : (F, 24, 3) float32 — corrected world-space joint positions
    root_trans   : (F, 3)     float32 — pelvis/root world translation per frame
    fps          : frames per second
    """
    joints_world: np.ndarray  # (F, 24, 3)
    root_trans: np.ndarray    # (F, 3)
    fps: float

