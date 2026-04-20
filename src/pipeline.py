"""
pipeline.py
Orchestrator — wires all 7 stages end-to-end.

Usage:
    from src.pipeline import run_pipeline
    run_pipeline(
        video_path       = "data/input.mp4",
        calibration_path = "data/calibration.npz",
        marker_length    = 0.20,
        output_bvh       = "data/output/animation.bvh",
        target_fps       = 30.0,
    )

Or with config defaults:
    import config
    from src.pipeline import run_pipeline
    run_pipeline(
        video_path       = str(config.INPUT_VIDEO_PATH),
        calibration_path = str(config.CALIBRATION_PATH),
        marker_length    = config.MARKER_LENGTH,
        output_bvh       = str(config.OUTPUT_BVH_PATH),
    )
"""
import numpy as np

import config
from src.calibration import load_calibration
from src.stages.s1_extract_frames  import extract_frames
from src.stages.s2_pose_estimation import estimate_smpl
from src.stages.s3_aruco_transform import estimate_camera_world_transform
from src.stages.s4_world_joints    import to_world_joints
from src.stages.s5_contact_labels  import detect_foot_contacts
from src.stages.s6_ik_correction   import apply_foot_ik
from src.stages.s7_export_bvh      import export_bvh


def _log_stage(name: str, obj: object) -> None:
    """Log key shape information at each stage boundary."""
    attrs = vars(obj) if hasattr(obj, "__dict__") else {}
    shape_info = {
        k: v.shape if isinstance(v, np.ndarray) else v
        for k, v in attrs.items()
    }
    print(f"[Pipeline] {name}: {shape_info}")


def run_pipeline(
    video_path: str = str(config.INPUT_VIDEO_PATH),
    calibration_path: str = str(config.CALIBRATION_PATH),
    marker_length: float = config.MARKER_LENGTH,
    output_bvh: str = str(config.OUTPUT_BVH_PATH),
    target_fps: float = config.TARGET_FPS,
    device: str = config.DEVICE,
    height_thresh: float = config.HEIGHT_THRESH,
    vel_thresh: float = config.VEL_THRESH,
) -> None:
    """
    Run the full Ani-Me pipeline from video to BVH.

    Parameters
    ----------
    video_path        : path to input video file
    calibration_path  : path to .npz camera calibration file
    marker_length     : physical side length of ArUco markers in metres
    output_bvh        : destination path for the output .bvh file
    target_fps        : extraction framerate (default 30)
    device            : torch device for ROMP ('cuda' or 'cpu')
    height_thresh     : foot contact height threshold in metres
    vel_thresh        : foot contact velocity threshold in metres/frame
    """
    print("[Pipeline] Starting Ani-Me pipeline")

    # Stage 0: Load camera calibration
    print("[Pipeline] Stage 0 — Loading calibration")
    camera_matrix, dist_coeffs = load_calibration(calibration_path)

    # Stage 1: Extract frames
    print("[Pipeline] Stage 1 — Extracting frames")
    frames = extract_frames(video_path, target_fps=target_fps)
    _log_stage("FrameSequence", frames)

    # Stage 2: SMPL pose estimation
    print("[Pipeline] Stage 2 — Estimating SMPL pose")
    smpl = estimate_smpl(frames, device=device)
    _log_stage("SMPLSequence", smpl)

    # Stage 3: Camera → world transform via ArUco
    print("[Pipeline] Stage 3 — Estimating camera-to-world transform")
    transform = estimate_camera_world_transform(
        frames, marker_length, camera_matrix, dist_coeffs
    )
    _log_stage("CameraWorldTransform", transform)

    # Stage 4: Project joints to world space
    print("[Pipeline] Stage 4 — Projecting joints to world space")
    world_joints = to_world_joints(smpl, transform)
    world_joints.fps = target_fps   # fill fps sentinel
    _log_stage("WorldJointSequence", world_joints)

    # Stage 5: Foot contact detection
    print("[Pipeline] Stage 5 — Detecting foot contacts")
    contacts = detect_foot_contacts(
        world_joints,
        height_thresh=height_thresh,
        vel_thresh=vel_thresh,
    )
    left_contacts  = contacts.labels[:, 0].sum()
    right_contacts = contacts.labels[:, 1].sum()
    print(f"[Pipeline]   Left foot contact frames:  {left_contacts}")
    print(f"[Pipeline]   Right foot contact frames: {right_contacts}")

    # Stage 6: IK correction
    print("[Pipeline] Stage 6 — Applying foot IK correction")
    ik_joints = apply_foot_ik(world_joints, contacts)
    _log_stage("IKJointSequence", ik_joints)

    # Stage 7: Export BVH
    print("[Pipeline] Stage 7 — Exporting BVH")
    export_bvh(ik_joints, output_bvh)

    print(f"[Pipeline] Done. Output: {output_bvh}")

