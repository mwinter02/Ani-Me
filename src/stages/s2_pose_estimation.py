"""
s2_pose_estimation.py
FrameSequence → SMPLSequence

Runs ROMP_Simple on each frame to extract SMPL pose (theta) and shape (beta)
parameters, along with joint positions in camera space. A single person per
frame is assumed. Beta is taken as the median across all frames to provide
stable, consistent limb lengths for IK downstream.
"""
import numpy as np

from src.types import FrameSequence, SMPLSequence


def estimate_smpl(frames: FrameSequence, device: str = "cuda") -> SMPLSequence:
    """
    Estimate per-frame SMPL parameters from a frame sequence using ROMP_Simple.

    Parameters
    ----------
    frames : FrameSequence
        RGB frame sequence from Stage 1.
    device : str
        Torch device to run ROMP_Simple on ('cuda' or 'cpu').

    Returns
    -------
    SMPLSequence
        theta      : (F, 72)    float32 — per-frame pose parameters
        beta       : (10,)      float32 — median body shape across all frames
        joints_cam : (F, 24, 3) float32 — joint positions in camera space, metres
        n_frames   : F
    """
    import romp

    settings = romp.main.default_settings()
    settings.mode = "image"
    settings.show = False
    settings.show_largest = True   # single-person: use largest detected person
    settings.device = device

    model = romp.ROMP(settings)

    all_theta = []
    all_beta  = []
    all_joints = []

    for frame_rgb in frames.frames:
        # ROMP expects BGR
        import cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result = model(frame_bgr)

        if result is None or len(result) == 0:
            # Pad with zeros if detection fails for a frame
            all_theta.append(np.zeros(72, dtype=np.float32))
            all_beta.append(np.zeros(10, dtype=np.float32))
            all_joints.append(np.zeros((24, 3), dtype=np.float32))
            continue

        # Take the first (largest) detected person
        all_theta.append(result["smpl_thetas"][0].astype(np.float32))   # (72,)
        all_beta.append(result["smpl_betas"][0].astype(np.float32))     # (10,)
        all_joints.append(result["joints"][0, :24].astype(np.float32))  # (24, 3)

    theta      = np.stack(all_theta,  axis=0)   # (F, 72)
    betas_all  = np.stack(all_beta,   axis=0)   # (F, 10)
    joints_cam = np.stack(all_joints, axis=0)   # (F, 24, 3)

    # Median beta across frames for stable limb lengths
    beta = np.median(betas_all, axis=0)         # (10,)

    return SMPLSequence(
        theta=theta,
        beta=beta,
        joints_cam=joints_cam,
        n_frames=len(frames.frames),
    )

