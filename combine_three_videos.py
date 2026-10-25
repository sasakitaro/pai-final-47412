"""
Combines the 3 simulation videos at 2x speed with the opening/ending slide image:
- Opening (4.0s) : PAI応用1_最終課題_スライド.jpg
- Part 1 (37.5s) : PAI最終課題_47412.mp4          (2x Speed - Basic Color Sorting)
- Part 2 (37.5s) : PAI最終課題_47412_shuffled.mp4 (2x Speed - Shuffled Cross-Matching)
- Part 3 (37.5s) : PAI最終課題_47412_reject.mp4   (2x Speed - Mismatch Rejection & Return-to-Origin)
- Ending  (4.0s) : PAI応用1_最終課題_スライド.jpg
Total Duration   : ~120.5 seconds (2m 00.5s, strictly within the 1-3 minute requirement)
Output Filename  : PAI最終課題_47412_combined.mp4
"""

import os
import sys
import cv2
import numpy as np


def load_image_unicode(path: str) -> np.ndarray:
    """Load an image from a path containing Japanese characters on Windows/Linux."""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return img


def draw_2x_badge(frame_bgr: np.ndarray, part_label: str) -> np.ndarray:
    """Draw a '2x Speed' indicator badge in the top-right corner as required by the assignment guidelines."""
    out = frame_bgr.copy()
    h, w, _ = out.shape
    box_w, box_h = 235, 32
    x1, y1 = w - box_w - 14, 18
    x2, y2 = w - 14, 18 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 25, 40), -1)
    out = cv2.addWeighted(overlay, 0.82, out, 0.18, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (80, 220, 255), 1)

    cv2.putText(
        out,
        f"{part_label} | 2x Speed",
        (x1 + 10, y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (120, 255, 190),
        1,
        cv2.LINE_AA,
    )
    return out


def combine_videos(
    slide_path: str = "PAI応用1_最終課題_スライド.jpg",
    video1_path: str = "PAI最終課題_47412.mp4",
    video2_path: str = "PAI最終課題_47412_shuffled.mp4",
    video3_path: str = "PAI最終課題_47412_reject.mp4",
    output_path: str = "PAI最終課題_47412_combined.mp4",
    slide_duration_sec: float = 4.0,
) -> str:
    width, height = 960, 540
    fps = 30

    for p in [slide_path, video1_path, video2_path, video3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input file not found: {os.path.abspath(p)}")

    slide_bgr = load_image_unicode(slide_path)
    slide_bgr = cv2.resize(slide_bgr, (width, height), interpolation=cv2.INTER_AREA)

    temp_out = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    slide_frames = int(round(slide_duration_sec * fps))

    # 1. Opening slide
    for _ in range(slide_frames):
        writer.write(slide_bgr)

    # 2. Append each of the 3 videos at 2x speed (every 2nd frame)
    parts = [
        (video1_path, "Part 1/3: Aligned"),
        (video2_path, "Part 2/3: Shuffled"),
        (video3_path, "Part 3/3: Reject"),
    ]

    total_written = slide_frames
    for v_path, label in parts:
        cap = cv2.VideoCapture(v_path)
        frame_idx = 0
        part_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 2 == 0:
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                badged = draw_2x_badge(frame, label)
                writer.write(badged)
                part_frames += 1
                total_written += 1
            frame_idx += 1
        cap.release()
        print(f"Added {v_path} at 2x speed: {part_frames} frames ({part_frames / fps:.1f}s)")

    # 3. Ending slide
    for _ in range(slide_frames):
        writer.write(slide_bgr)
    total_written += slide_frames

    writer.release()

    # Re-encode with ffmpeg for H.264 / YouTube compatibility
    cmd = f'ffmpeg -y -loglevel error -i "{temp_out}" -vcodec libx264 -pix_fmt yuv420p -movflags +faststart "{output_path}"'
    if os.system(cmd) == 0:
        try:
            os.remove(temp_out)
        except OSError:
            pass
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_out, output_path)

    total_sec = total_written / float(fps)
    print(f"[SUCCESS] Combined video saved to: {os.path.abspath(output_path)}")
    print(f"Total frames: {total_written} | Total duration: {total_sec:.1f}s ({int(total_sec//60)}m {total_sec%60:.1f}s)")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    combine_videos()
