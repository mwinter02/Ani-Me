"""
config.py
Central configuration — all default file paths and pipeline parameters.

Import this anywhere in the project:
    import config
    config.CALIBRATION_PATH
    config.TARGET_FPS
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Project root (directory this file lives in)
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Directories ───────────────────────────────
DATA_PATH           = ROOT / "data"
FRAMES_PATH         = ROOT / "data" / "frames"
OUTPUT_PATH         = ROOT / "data" / "output"

# ── Inputs ────────────────────────────────────
CALIBRATION_PATH        = ROOT / "data" / "calibration.npz"
INPUT_VIDEO_PATH        = ROOT / "data" / "input.mp4"
CALIBRATION_GRID_PATH   = ROOT / "calibration_grid.jpg"

# Checkerboard dimensions (inner corners, not squares)
# Matches the bundled calibration_grid.jpg
# A 6 cols x 9 rows grid (7x10 squares) works well for a portrait phone screen
GRID_ROWS               = 9
GRID_COLS               = 6

# ── Outputs ───────────────────────────────────
OUTPUT_BVH_PATH     = ROOT / "data" / "output" / "animation.bvh"

# ─────────────────────────────────────────────
# Pipeline parameters
# ─────────────────────────────────────────────

# Stage 1 — Frame extraction
TARGET_FPS          = 30.0

# Stage 2 — Pose estimation
DEVICE              = "cpu"         # "cuda" or "mps" if available

# Stage 3 — ArUco transform
MARKER_LENGTH       = 0.20          # Physical side length of ArUco marker in metres
ARUCO_DICT          = "DICT_4X4_50"

# Stage 5 — Foot contact detection
HEIGHT_THRESH       = 0.05          # Max foot z-height (metres) to label as grounded
VEL_THRESH          = 0.20          # Max foot speed (metres/frame) to label as stationary

