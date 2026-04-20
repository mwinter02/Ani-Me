# Setup Guide

## Environment Setup
> **Platform:** `setup.sh` and `prepare_smpl.sh` are tested on **macOS** with Python 3.12 installed
> via [python.org](https://www.python.org/downloads/). They should work on Linux with minor adjustments (the SSL
> certificate fix step is macOS-only and will be skipped automatically). Windows is not supported by the scripts — use
> WSL2.

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:

1. Create a `.venv` Python virtual environment
2. Install all dependencies (numpy, OpenCV, PyTorch, simple_romp, smplx, etc.)
3. Print step-by-step instructions for the manual SMPL model file setup below

> **Note:** SMPL model files must be set up manually before running the pipeline (licensing prevents automation).

---

### SMPL Model File Setup

**1. Download ROMP metadata**
[smpl_model_data.zip](https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip) — unzip to get a
folder named `smpl_model_data/`

**2. Download SMPL model files** *(free account required)*
[SMPL_python_v.1.1.0.zip](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip) — after
unzipping, locate:

```
smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

Copy it into your `smpl_model_data/` folder and **rename it to `SMPL_NEUTRAL.pkl`**.

Your folder should look like:

```
smpl_model_data/
├── SMPL_NEUTRAL.pkl          ← renamed from basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
└── smpl_kid_template.npy
```

**3. Convert for ROMP**

```bash
romp.prepare_smpl -source_dir=/path/to/smpl_model_data
```

This saves converted files to `~/.romp/`. You should see:

```
~/.romp/
├── SMPL_NEUTRAL.pth
└── SMPLA_NEUTRAL.pth
```

---

## Camera Calibration


The pipeline requires camera intrinsics (focal length, principal point, distortion) to accurately resolve the ArUco marker pose. This is a **one-time step per camera** — once you have a `calibration.npz` you can reuse it for every recording session as long as you don't change the camera's zoom or resolution.

#### What you need

The bundled `calibration_grid.jpg` is a **ChArUco board** — a chessboard with ArUco markers embedded in the white squares. It is more robust than a plain chessboard: it works even when partially occluded and at more extreme angles.

Board spec: **7 × 10 squares** (6 × 9 inner corners), dictionary `DICT_4X4_50`, portrait orientation.

Display it on your phone at full brightness. No printing required — physical size does not matter.

To generate an alternative board: [https://calib.io/pages/camera-calibration-pattern-generator](https://calib.io/pages/camera-calibration-pattern-generator)
Select **Target Type = ChArUco**, **Dictionary = DICT_4X4**, update `GRID_COLS`/`GRID_ROWS` in `config.py` to match.

**Tips:**
- Set the phone screen to full brightness
- The board should fill most of the webcam frame — bring it close
- Keep the laptop still on a desk and move the phone in front of it
- Partial occlusion is fine — only a few markers need to be visible at a time
- Tilt the phone to vary the angle; avoid always holding it perfectly front-on

#### Option A — Live capture from webcam (recommended)

```bash
source .venv/bin/activate
python calibrate_camera.py --output data/calibration.npz
```

A window will open showing your webcam feed. Set your laptop on a desk and move your phone (displaying the checkerboard) in front of it at different positions and angles. The terminal will prompt you where to position your phone. Once your phone is ready, press SPACE to queue a capture. You may need to move your phone slightly around to get it to detect properly.

Once it has captured, a new instruction will display in the terminal. Follow and repeat until finished.

#### Option B — From a folder of images

If you prefer to pre-capture images (e.g. from your phone or a separate camera app):

```bash
python calibrate_camera.py --images ./calib_images/ --output calibration.npz
```

#### Custom checkerboard size

If your printed board has different dimensions, specify the number of **inner corners** (not squares):

```bash
python calibrate_camera.py --cols 9 --rows 6 --output calibration.npz
```

#### Checking quality

The script reports a **reprojection error** in pixels after calibration. A value below **1.0 px** is good. If it's higher, try recapturing with better lighting, a flatter board, or more varied angles.

```
[calibrate] RMS reprojection error: 0.412 px  ← good
[calibrate] RMS reprojection error: 2.1 px    ← recapture recommended
```

#### Important

- Use the **same camera, same resolution, and same zoom/focal length** that you'll use to record your motion video
- If you change any of these, recalibrate
- Keep `calibration.npz` in `data/` (the pipeline looks for it there by default)

---


