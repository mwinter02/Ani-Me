"""
calibrate_camera.py
One-time camera calibration using a ChArUco board pattern.

ChArUco (Chessboard + ArUco) is more robust than a standard chessboard —
it works even when the board is partially occluded or at extreme angles,
because each ArUco marker can be identified independently.

Usage:
    # Live capture from webcam (press SPACE to capture a frame, Q to finish)
    python calibrate_camera.py

    # From a folder of pre-captured images
    python calibrate_camera.py --images ./calib_images/

    # Override output path
    python calibrate_camera.py --output data/calibration.npz

ChArUco board:
    The bundled calibration_grid.jpg is a ChArUco board with:
      - 7 cols x 10 rows of squares  (6 x 9 inner corners)
      - ArUco dictionary: DICT_4X4_50
    Generate alternative boards at: https://calib.io/pages/camera-calibration-pattern-generator
    Select: Target Type = ChArUco, Dictionary = DICT_4X4

    Tips:
      - Display calibration_grid.jpg on your phone at full brightness
      - The board should fill most of the webcam frame — bring it close
      - Keep the laptop still on a desk and move the phone in front of it
      - Partial occlusion is fine — ChArUco only needs a few visible markers
      - Tilt the phone to vary the angle; avoid holding it perfectly flat/frontal
"""

import argparse
import glob
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

import config


# ─────────────────────────────────────────────
# Build the ChArUco board from config parameters
# ─────────────────────────────────────────────
def _build_board() -> tuple:
    """Return (board, charuco_detector)."""
    aruco_dict = aruco.getPredefinedDictionary(
        getattr(aruco, config.ARUCO_DICT)
    )
    # squares = inner_corners + 1
    board = aruco.CharucoBoard(
        (config.GRID_COLS + 1, config.GRID_ROWS + 1),
        squareLength=1.0,
        markerLength=0.8,
        dictionary=aruco_dict,
    )
    charuco_detector = aruco.CharucoDetector(board)
    return board, charuco_detector


def _detect_charuco(gray: np.ndarray, charuco_detector):
    """
    Detect ChArUco corners in a grayscale frame using the OpenCV 4.7+ API.
    Returns (charuco_corners, charuco_ids, n_corners) or (None, None, 0).
    """
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None, 0
    return charuco_corners, charuco_ids, len(charuco_ids)


def collect_frames_live(target: int = 20) -> list[np.ndarray]:
    """
    SPACE-triggered capture: press SPACE to queue a capture, which fires
    automatically on the next frame where the board is detected.
    The preview is mirrored for natural interaction; raw frames are saved.
    Stops automatically once `target` frames are captured.
    """
    board, charuco_detector = _build_board()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[calibrate] ERROR: Cannot open webcam.")
        sys.exit(1)

    frames = []
    capture_queued = False
    banner_until = 0.0

    print("[calibrate] Calibration capture — ChArUco board detection.")
    print(f"[calibrate] Display {config.CALIBRATION_GRID_PATH.name} on your phone at full brightness.")
    print("[calibrate] Press SPACE to queue a capture — it fires when the board is next detected.")
    print(f"[calibrate] Target: {target} frames. Press Q to finish early (need at least 6).")

    POSES = [
        "flat, board filling frame",
        "tilt LEFT ~30°",
        "tilt RIGHT ~30°",
        "tilt UP ~30°",
        "tilt DOWN ~30°",
        "top-left corner of frame",
        "top-right corner of frame",
        "bottom-left corner of frame",
        "bottom-right corner of frame",
        "diagonal tilt — top-left lean",
        "diagonal tilt — top-right lean",
        "closer to camera",
        "further from camera",
        "tilt LEFT ~45°",
        "tilt RIGHT ~45°",
        "tilt UP ~45°",
        "tilt DOWN ~45°",
        "roll the phone clockwise ~20°",
        "roll the phone anti-clockwise ~20°",
        "flat again, centred",
    ]

    while len(frames) < target:
        ret, frame = cap.read()
        if not ret:
            break

        now = cv2.getTickCount() / cv2.getTickFrequency()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, n = _detect_charuco(gray, charuco_detector)

        # Fire queued capture on next detected frame
        if capture_queued and charuco_corners is not None:
            frames.append(frame.copy())   # save raw (unmirrored) frame
            capture_queued = False
            banner_until = now + 0.8
            idx = len(frames)
            next_pose = POSES[idx] if idx < len(POSES) else "varied angle"
            print(f"[calibrate] Captured {idx}/{target} — next: {next_pose}")

        # Mirror display only — raw frame is saved above
        display = cv2.flip(frame, 1)

        if now < banner_until:
            cv2.rectangle(display, (0, 0), (display.shape[1], 55), (0, 180, 0), -1)
            cv2.putText(display, "Captured! — move/tilt the phone, then press SPACE",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif capture_queued:
            if charuco_corners is not None:
                # Mirror the corners for display only
                display_corners = charuco_corners.copy()
                display_corners[:, 0, 0] = display.shape[1] - charuco_corners[:, 0, 0]
                aruco.drawDetectedCornersCharuco(display, display_corners, charuco_ids)
                cv2.putText(display, "Board detected — capturing now...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            else:
                cv2.putText(display, "Capture queued — waiting for board detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        elif charuco_corners is not None:
            display_corners = charuco_corners.copy()
            display_corners[:, 0, 0] = display.shape[1] - charuco_corners[:, 0, 0]
            aruco.drawDetectedCornersCharuco(display, display_corners, charuco_ids)
            cv2.putText(display, f"Board detected ({n} corners) — press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Board not detected — bring closer or adjust angle",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Next pose hint
        next_idx = len(frames)
        if next_idx < len(POSES) and now >= banner_until:
            cv2.putText(display, f"Next pose: {POSES[next_idx]}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        # Progress bar
        progress = int(display.shape[1] * len(frames) / target)
        cv2.rectangle(display, (0, display.shape[0] - 14), (display.shape[1], display.shape[0]), (50, 50, 50), -1)
        cv2.rectangle(display, (0, display.shape[0] - 14), (progress, display.shape[0]), (0, 200, 80), -1)
        cv2.putText(display, f"{len(frames)}/{target} frames  |  SPACE=capture  Q=done",
                    (10, display.shape[0] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capture_queued = True
            print("[calibrate] Capture queued — move to next pose and hold steady")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) == target:
        print(f"[calibrate] Target reached — {len(frames)} frames captured.")
    else:
        print(f"[calibrate] Stopped early — {len(frames)} frames captured.")

    return frames


def collect_frames_from_folder(path: str) -> list[np.ndarray]:
    """Load calibration frames from a folder of images."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(path, p)))
    image_paths.sort()

    if not image_paths:
        print(f"[calibrate] ERROR: No images found in {path}")
        sys.exit(1)

    print(f"[calibrate] Found {len(image_paths)} images in {path}")
    frames = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    return frames


def calibrate(frames: list[np.ndarray], output_path: str) -> None:
    """Run ChArUco calibration on captured frames and save results to .npz."""
    if len(frames) < 6:
        print(f"[calibrate] ERROR: Need at least 6 frames, got {len(frames)}.")
        sys.exit(1)

    board, charuco_detector = _build_board()

    all_charuco_corners = []
    all_charuco_ids = []
    used = 0

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, n = _detect_charuco(gray, charuco_detector)
        if charuco_corners is not None:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            used += 1
        else:
            print(f"[calibrate] Frame {i+1}: board not detected — skipping.")

    if used < 6:
        print(f"[calibrate] ERROR: Board detected in only {used} frames.")
        print("[calibrate] Try better lighting or more varied angles.")
        sys.exit(1)

    print(f"[calibrate] Running ChArUco calibration on {used}/{len(frames)} frames...")

    # calibrateCameraCharuco was removed in OpenCV 4.7+.
    # The replacement: use board.matchImagePoints() to get 3D↔2D correspondences,
    # then pass them to the standard cv2.calibrateCamera.
    obj_points = []
    img_points = []
    for charuco_corners, charuco_ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        obj_points.append(obj_pts)
        img_points.append(img_pts)

    h, w = frames[0].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    print(f"[calibrate] Calibration complete.")
    print(f"[calibrate] RMS reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("[calibrate] WARNING: reprojection error > 1.0 px — consider recapturing "
              "with more varied angles and distances.")

    print(f"[calibrate] Camera matrix:\n{camera_matrix}")
    print(f"[calibrate] Distortion coefficients: {dist_coeffs.flatten()}")

    np.savez(output_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"[calibrate] Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate camera using a ChArUco board."
    )
    parser.add_argument(
        "--images", type=str, default=None,
        help="Path to folder of calibration images. If omitted, uses live auto-capture."
    )
    parser.add_argument(
        "--output", type=str, default=str(config.CALIBRATION_PATH),
        help=f"Output path for the calibration .npz file (default: {config.CALIBRATION_PATH})"
    )
    parser.add_argument(
        "--target", type=int, default=20,
        help="Number of frames to capture before stopping (default: 20)"
    )
    args = parser.parse_args()

    if args.images:
        frames = collect_frames_from_folder(args.images)
    else:
        frames = collect_frames_live(target=args.target)

    calibrate(frames, args.output)


if __name__ == "__main__":
    main()

