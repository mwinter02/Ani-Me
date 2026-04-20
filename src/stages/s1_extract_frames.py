"""
s1_extract_frames.py
Video → FrameSequence

Decodes the input video at a fixed target FPS using OpenCV.
All downstream stages (ROMP and ArUco) consume the same FrameSequence,
guaranteeing frame-perfect synchronisation and a known Δt for velocity
calculations.
"""
import cv2
import numpy as np

from src.types import FrameSequence


def extract_frames(video_path: str, target_fps: float = 30.0) -> FrameSequence:
    """
    Decode a video file into a fixed-rate RGB frame sequence.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    target_fps : float
        Target extraction framerate. Frames are sampled from the source video
        at this rate. Explicit fixed FPS avoids VFR issues and ensures Δt is
        known for velocity calculations downstream.

    Returns
    -------
    FrameSequence
        frames      : (F, H, W, 3) uint8 RGB
        fps         : target_fps
        source_path : video_path
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise ValueError(f"Could not read FPS from video: {video_path}")

    # Sample every Nth source frame to approximate target_fps
    frame_interval = max(1, round(source_fps / target_fps))

    frames = []
    source_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if source_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        source_idx += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    return FrameSequence(
        frames=np.stack(frames, axis=0),  # (F, H, W, 3)
        fps=target_fps,
        source_path=video_path,
    )

