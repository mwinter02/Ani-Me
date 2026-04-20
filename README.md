# Ani-Me

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

# TODO

- [x] Camera calibration script
- [ ] Pelvis height adjustment in IK - Stage 6 moves ankle/knee but doesn't yet push the root up/down to compensate
- [ ] BVH using SMPL theta rotations - quaternions? Stage 7 currently infers rotations from positions; should use theta from SMPLSequence directly
- [ ] All testing files


  - [ ] s1_extract_frames.py — Low risk
Frame sampling logic is straightforward. Likely works as-is. Needs a real video to confirm.

  - [ ] s2_pose_estimation.py — Medium risk
Calls ROMP's API. The exact output keys (smpl_thetas, smpl_betas, joints) need to be verified against what simple_romp 1.1.4 actually returns — ROMP's API has changed across versions and the key names in the code may need adjusting.

  - [ ] s3_aruco_transform.py — Medium risk
ArUco detection and solvePnP logic is standard OpenCV. The rotation averaging via SVD is correct in theory. Needs a real frame with markers to validate the transform is sensible (right scale, right orientation).

  - [ ] s4_world_joints.py — Low risk
Pure numpy matrix multiplication. Should work, but the world-space result needs visual sanity checking — is the person the right height above the floor? Are axes oriented correctly?

  - [ ] s5_contact_labels.py — Medium risk
The thresholds (height_thresh=0.05m, vel_thresh=0.20m/frame) are reasonable starting values but will almost certainly need tuning once you see real ROMP output. The z-axis assumption (z=0 is floor) also depends on stage 4 being correct.

  - [ ] s6_ik_correction.py — High risk
This is the most algorithmically complex stage. The two-bone IK solver is implemented from scratch. Known risks:
Knee flip detection may not be robust enough in all poses
The pelvis height adjustment is not yet implemented (noted in Overview as needed but the current code just moves knee/ankle without adjusting root)
Contact event blending (smooth in/out) is not implemented — hard lock only

  - [ ] s7_export_bvh.py — High risk
BVH export is notoriously finicky. The local Euler angle computation from joint position deltas is a simplified approximation — a proper BVH writer should use the SMPL theta rotation parameters directly rather than inferring rotations from position changes. This will likely produce a valid BVH file structurally but with rotation artifacts.