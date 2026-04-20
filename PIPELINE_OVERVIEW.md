# Ani-Me — Pipeline Overview

## Overview
We aim to build a pipeline that converts human motion from video into 3D character animation. Given a video of a person
performing an action, we will:

- Reconstruct a 3D pose sequence (e.g. SMPL) using a pose estimation model (e.g. ROMP)
- Estimate the ground plane and detect foot contacts using computer vision techniques and temporal reasoning
- Apply inverse kinematics (IK) optimization to correct foot sliding and ensure realistic foot-ground interactions
- Output a corrected motion sequence that can be retargeted to a target skeleton and exported as an animation file or
  Blender scene

```
Video file
  ↓
SMPL sequence (via ROMP)
  ↓
ground plane estimation (CV geometry)
  ↓
contact detection (temporal reasoning)
  ↓
IK optimization (constraints)
  ↓
corrected motion sequence
  ↓
export (FBX, BVH, Blender)
```

## Project Structure

```
ani-me/
├── anime/
│   ├── __init__.py
│   ├── types.py                  # shared dataclasses and joint index constants
│   ├── pipeline.py               # orchestrator — wires all stages end-to-end
│   ├── calibration.py            # load camera intrinsics from calibration file
│   └── stages/
│       ├── __init__.py
│       ├── s1_extract_frames.py  # Video → FrameSequence
│       ├── s2_pose_estimation.py # FrameSequence → SMPLSequence
│       ├── s3_aruco_transform.py # FrameSequence → CameraWorldTransform
│       ├── s4_world_joints.py    # SMPLSequence + Transform → WorldJointSequence
│       ├── s5_contact_labels.py  # WorldJointSequence → ContactLabelSequence
│       ├── s6_ik_correction.py   # WorldJoints + ContactLabels → IKJointSequence
│       └── s7_export_bvh.py      # IKJointSequence → BVH file
├── tests/
├── requirements.txt
└── README.md
```

---

## Shared Types — `anime/types.py`

| Dataclass              | Key Fields                          | Shape / Units                                               |
| ---------------------- | ----------------------------------- | ----------------------------------------------------------- |
| `FrameSequence`        | `frames`, `fps`, `source_path`      | `(F, H, W, 3)` uint8 RGB, fps in Hz                         |
| `SMPLSequence`         | `theta`, `beta`, `joints_cam`       | `(F, 72)`, `(10,)`, `(F, 24, 3)` — camera space, metres     |
| `CameraWorldTransform` | `T_cw`, `locked_frame_idx`          | `(4, 4)` float32 — z=0 is floor                             |
| `WorldJointSequence`   | `joints_world`, `fps`               | `(F, 24, 3)` — world space, z-up, metres                    |
| `ContactLabelSequence` | `labels`, `fps`                     | `(F, 2)` bool — `[left_foot, right_foot]`                   |
| `IKJointSequence`      | `joints_world`, `root_trans`, `fps` | `(F, 24, 3)`, `(F, 3)` — root translation separated for BVH |

Foot joint index constants: `LEFT_ANKLE=7`, `LEFT_TOE=10`, `RIGHT_ANKLE=8`, `RIGHT_TOE=11`

---

## Pipeline Stages

---

### Stage 0: Camera Calibration (Prerequisite)

**`calibration.py`**
```
load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]
  path     : path to .npz calibration file (produced by OpenCV checkerboard calibration)
  returns  : (camera_matrix (3,3), dist_coeffs (5,))
```

---

### Stage 1: Video → Frame Sequence

**`s1_extract_frames.py`**
```
extract_frames(video_path: str, target_fps: float = 30.0) -> FrameSequence
  video_path  : path to input video file
  target_fps  : desired extraction framerate — explicit fixed FPS avoids VFR issues
                and ensures Δt is known for velocity calculations downstream
  returns     : FrameSequence — (F, H, W, 3) RGB frames + fps
```

Both ROMP and OpenCV read from this same frame sequence, guaranteeing frame-perfect synchronisation. The video is decoded exactly once.

---

### Stage 2: SMPL Pose Estimation

**`s2_pose_estimation.py`**
```
estimate_smpl(frames: FrameSequence, device: str = "cuda") -> SMPLSequence
  frames   : FrameSequence from Stage 1
  device   : torch device for ROMP_Simple
  returns  : SMPLSequence — per-frame theta (F, 72), median beta (10,),
             joints in camera space (F, 24, 3) in metres
```

`beta` is the **median across all frames** to give stable, consistent limb lengths for IK downstream.

**Base:**
Process frames through ROMP_Simple to extract per-frame SMPL parameters (pose θ, shape β) and joint positions in camera space. Assume a single person in frame.

**Extension:**
- Switch to a temporally-aware model (e.g. WHAM or VIBE) for smoother pose sequences
- Handle multi-person scenes
- Use video-mode inference instead of per-frame for better temporal coherence

---

### Stage 3: Camera → World Space Transform

**`s3_aruco_transform.py`**
```
estimate_camera_world_transform(
    frames         : FrameSequence,
    marker_length  : float,
    camera_matrix  : np.ndarray,
    dist_coeffs    : np.ndarray
) -> CameraWorldTransform
  marker_length  : physical size of printed ArUco marker in metres
  camera_matrix  : (3, 3) intrinsics from calibration
  dist_coeffs    : (5,) distortion coefficients from calibration
  returns        : CameraWorldTransform — locked (4, 4) T_cw, index of anchor frame
```

Iterates frames until any marker is detected. Averages pose estimates across all visible markers in that frame. Locks the result for the entire sequence (static camera base case).

**Base:**
Static camera with **multiple ArUco markers** placed flat on the floor. Any visible marker in the first detected frame anchors the transform for the whole sequence. The marker coordinate frame defines the world origin and floor plane (`z = 0`).

**Extension:**
- Dynamic camera: compute transform per-frame, interpolate (lerp + slerp) across fully occluded frames
- Markerless fallback: RANSAC fit to lowest mesh vertices across the sequence

---

### Stage 4: Ground Plane Estimation

**`s4_world_joints.py`** (handled implicitly by the ArUco transform)
```
to_world_joints(
    smpl       : SMPLSequence,
    transform  : CameraWorldTransform
) -> WorldJointSequence
  smpl       : SMPLSequence from Stage 2
  transform  : CameraWorldTransform from Stage 3
  returns    : WorldJointSequence — joints (F, 24, 3) in world space, z=0 is floor
```

Homogeneous-extends `joints_cam` to `(F, 24, 4)`, left-multiplies by `T_cw`, strips the homogeneous column.

**Base:**
Ground plane is **already defined** by the ArUco marker frame — `z = 0` is the floor by construction. No additional fitting needed. Foot joint heights serve as a sanity check.

**Extension:**
- For the markerless case: RANSAC fit to lowest heel/toe vertex positions, reorient so that plane becomes `z = 0`
- Warn when foot joints consistently penetrate the plane (indicates scale or calibration error)

---

### Stage 5: Foot Contact Detection

**`s5_contact_labels.py`**
```
detect_foot_contacts(
    world_joints   : WorldJointSequence,
    height_thresh  : float = 0.05,
    vel_thresh     : float = 0.20
) -> ContactLabelSequence
  height_thresh  : max z-height in metres to consider a foot grounded
  vel_thresh     : max joint velocity in metres/frame to consider a foot stationary
  returns        : ContactLabelSequence — (F, 2) bool [left_foot, right_foot]
```

Both conditions must be simultaneously true. Uses both ankle AND toe joints — contact is labelled true if either is within threshold. `Δt = 1 / fps` is taken from `WorldJointSequence.fps`.

**Base:**
Per-frame binary classification using height threshold and velocity threshold on world-space joint positions.

**Extension:**
- Temporal smoothing via sliding window or median filter to remove flickering labels
- Hysteresis to prevent rapid toggling at contact boundaries
- Toe vs. heel distinction for heel-strike / toe-off phase detection
- Soft contact probability score rather than binary label

---

### Stage 6: IK Optimization — Foot Sliding Correction

**`s6_ik_correction.py`**
```
apply_foot_ik(
    world_joints  : WorldJointSequence,
    contacts      : ContactLabelSequence
) -> IKJointSequence
  world_joints  : WorldJointSequence from Stage 4
  contacts      : ContactLabelSequence from Stage 5
  returns       : IKJointSequence — corrected joints (F, 24, 3),
                  root translation (F, 3) separated out
```

Locks foot target to its world-space position at first contact frame. Solves hip→knee→ankle two-bone IK analytically (closed-form) per contact frame. Adjusts pelvis height to keep the limb within its natural length.

**Base:**
Analytical two-bone IK with hard foot lock for contact frames.

**Extension:**
- Full-body CCD or FABRIK optimization minimising deviation from original ROMP pose
- Soft constraints weighting foot lock against original pose
- Knee direction hinting to prevent knee-flipping
- Smooth blending into and out of contact events to avoid popping

---

### Stage 7: Export

**`s7_export_bvh.py`**
```
export_bvh(
    ik_joints     : IKJointSequence,
    output_path   : str,
    skeleton_def  : dict
) -> None
  ik_joints     : IKJointSequence from Stage 6
  output_path   : destination .bvh file path
  skeleton_def  : joint hierarchy (names, parent indices, rest-pose offsets)
```

Maps SMPL joint order to BVH hierarchy via `skeleton_def`. Converts world-space joint positions to local-frame Euler angles per frame. Writes `HIERARCHY` + `MOTION` sections as plain text.

**Base:**
BVH file output mapping SMPL joints to a standard BVH skeleton.

**Extension:**
- FBX export for game engines (Unreal, Unity)
- Blender Python script generation for direct scene import
- Skeleton retargeting to Mixamo / UE5 Mannequin via joint mapping table before export

---

### Orchestrator — `pipeline.py`
```
run_pipeline(
    video_path        : str,
    calibration_path  : str,
    marker_length     : float,
    output_bvh        : str,
    target_fps        : float = 30.0
) -> None
```

Calls all 7 stages in order, passing each output to the next. Logs data shapes at every stage boundary for debugging.

---

## Summary

| Stage                | Module                  | Base                               | Extension                                |
| -------------------- | ----------------------- | ---------------------------------- | ---------------------------------------- |
| Calibration          | `calibration.py`        | OpenCV checkerboard → .npz         | —                                        |
| 1. Frame Extraction  | `s1_extract_frames.py`  | ffmpeg/cv2, fixed FPS              | —                                        |
| 2. Pose Estimation   | `s2_pose_estimation.py` | ROMP_Simple, single person         | WHAM, multi-person                       |
| 3. Camera → World    | `s3_aruco_transform.py` | Static cam, multiple ArUco markers | Dynamic cam, slerp, RANSAC fallback      |
| 4. World Joints      | `s4_world_joints.py`    | Apply locked T_cw to joints        | —                                        |
| 5. Contact Detection | `s5_contact_labels.py`  | Height + velocity threshold        | Smoothing, hysteresis, heel/toe phases   |
| 6. IK Correction     | `s6_ik_correction.py`   | Analytical two-bone IK             | Full-body optimisation, soft constraints |
| 7. Export            | `s7_export_bvh.py`      | BVH                                | FBX, Blender, retargeting                |
