"""Video frame extraction utilities for dataset creation."""

import os

import cv2


def extract_frames(video_path, output_dir, max_frames=90, start_index=0, prefix="frame"):
    """Extract frames from a video file.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        max_frames: Maximum number of frames to extract per video.
        start_index: Starting index for frame numbering.
        prefix: Filename prefix for saved frames (e.g. 'rage', 'non_rage').

    Returns:
        Number of frames successfully extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0
    current_label = start_index

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_label += 1
        frame_path = os.path.join(output_dir, f"{prefix}_{current_label:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()
    return frame_count
