# FinalProject_Template

This repository contains templates for CSCI 1430 project deliverables of many kinds: proposals, progress checks, project reports, posters, etc.

At the beginnning of the project, you will submit a project proposal detailing the methods and data you plan on using in your final project.

A progress report will be due sometime in between the proposal and submission of the final project. This report will consist of accounts of work done by each individual team member so far on the project.

Finally, at the end of the project, you will submit a full project report. You will also complete an accompanying poster or presentation detailing your project, depending on the year. 

Feel free to look at individual files in this repository for more information. Good luck!


| Idea | Explanation | Resources | Related Work / Papers |
|------|------------|----------|------------------------|
| Physics-aware animation | Enforce physical realism (e.g., foot-ground contact, balance, gravity constraints) to avoid floating feet and unnatural motion | PyBullet, MuJoCo, Blender physics, inverse dynamics libraries | “Physics-Based Motion Capture with Human Dynamics” (arXiv:2303.18246) |
| Multi-person → multi-character animation | Extend pipeline to detect, track, and animate multiple people simultaneously, handling occlusion and identity tracking | OpenPose multi-person mode, DeepSORT (tracking), Detectron2 | AnimePose system (multi-person pipeline), multi-person pose estimation literature |
| Direct rig retargeting (no SMPL) | Skip SMPL model and directly map 2D/3D joints to Mixamo skeleton using inverse kinematics | Blender IK system, Unity Animation Rigging, custom IK solvers | Less explored; related to classical IK retargeting and game animation pipelines |
| Learning-based retargeting | Train a neural network to map pose keypoints directly to rig joint rotations instead of manual mapping | PyTorch, TensorFlow, motion capture datasets (Human3.6M, AMASS) | PoseNet3D, learned motion retargeting papers (various neural animation works) |
| Motion style transfer | Modify motion style (e.g., happy/sad/energetic walking) using latent representations or motion priors | Deep learning frameworks, motion datasets (CMU Mocap, AMASS) | Motion style transfer research, neural motion synthesis papers |
| Camera-aware reconstruction | Incorporate camera intrinsics/extrinsics to improve 3D pose accuracy and reduce projection errors | OpenCV (camera calibration), COLMAP, SfM tools | Camera-aware 3D pose estimation literature, monocular reconstruction papers |
| Robustness to occlusion | Predict missing joints when body parts are hidden using temporal or learned inference | Temporal CNNs, transformers, interpolation methods | AnimePose (handles occlusion), temporal pose estimation research |
| Real-time system | Build a live pipeline (webcam → pose → animation) with low latency for interactive applications | MediaPipe (fast), OpenPose GPU, Unity/Unreal Engine | Real-time pose tracking systems, AR/VR avatar animation pipelines |
