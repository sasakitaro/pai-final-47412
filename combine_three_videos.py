"""
Combines the 3 simulation videos at 2x speed with the opening/ending slide image
and Japanese Task Instruction (タスク指示文) banners:
- Opening (4.0s) : PAI応用1_最終課題_スライド.jpg
- Part 1 (37.5s) : PAI最終課題_47412.mp4          (2x Speed + 日本語タスク1指示文)
- Part 2 (37.5s) : PAI最終課題_47412_shuffled.mp4 (2x Speed + 日本語タスク2指示文)
- Part 3 (37.5s) : PAI最終課題_47412_reject.mp4   (2x Speed + 日本語タスク3指示文)
- Ending  (4.0s) : PAI応用1_最終課題_スライド.jpg
Total Duration   : ~120.5 seconds (2m 00.5s, strictly within the 1-3 minute requirement)
Output Filename  : PAI最終課題_47412_combined.mp4
"""

import os
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image_unicode(path: str) -> np.ndarray:
    """Load an image from a path containing Japanese characters on Windows/Linux."""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return img


def get_japanese_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """Load a Japanese TrueType font on Windows or Linux."""
    candidates = [
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\YuGothB.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_overlay_banner(
    width: int,
    part_badge: str,
    jp_instruction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-render the Japanese task instruction banner and 2x speed badge once per part
    for fast per-frame alpha blending.
    Returns (banner_bgr, alpha_mask_3ch) of shape (banner_h, width, 3).
    """
    banner_h = 36
    pil_img = Image.new("RGBA", (width, banner_h), (14, 18, 28, 225))
    draw = ImageDraw.Draw(pil_img)

    font_jp = get_japanese_font(14)
    font_badge = get_japanese_font(13)

    # Bottom accent line
    draw.line([(0, banner_h - 1), (width, banner_h - 1)], fill=(70, 195, 255, 255), width=2)

    # Japanese task instruction on left
    draw.text((12, 9), jp_instruction, font=font_jp, fill=(255, 240, 150, 255))

    # 2x Speed badge on right
    badge_x1, badge_y1 = width - 196, 4
    badge_x2, badge_y2 = width - 10, banner_h - 5
    draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=(24, 52, 78, 255), outline=(90, 225, 255, 255), width=1)
    draw.text((badge_x1 + 10, badge_y1 + 5), f"{part_badge} | 2倍速(2x)", font=font_badge, fill=(130, 255, 190, 255))

    rgba = np.array(pil_img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    alpha_3ch = np.repeat(alpha, 3, axis=2)
    return bgr, alpha_3ch


def apply_banner(frame_bgr: np.ndarray, banner_bgr: np.ndarray, alpha_3ch: np.ndarray, y_offset: int = 68) -> np.ndarray:
    """Blend the pre-rendered Japanese instruction banner onto the video frame right below the top header."""
    out = frame_bgr.copy()
    bh = banner_bgr.shape[0]
    roi = out[y_offset : y_offset + bh, :].astype(np.float32)
    blended = roi * (1.0 - alpha_3ch) + banner_bgr.astype(np.float32) * alpha_3ch
    out[y_offset : y_offset + bh, :] = np.clip(blended, 0, 255).astype(np.uint8)
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

    # 1. Opening slide (PAI応用1_最終課題_スライド.jpg)
    for _ in range(slide_frames):
        writer.write(slide_bgr)

    # 2. Append each of the 3 videos at 2x speed with Japanese task instruction banner
    parts = [
        (
            video1_path,
            "Part 1/3",
            "【タスク1 指示】3色のボックス(赤・緑・青)を認識し、同じ色の指定場所へ順番にPick & Placeせよ。",
        ),
        (
            video2_path,
            "Part 2/3",
            "【タスク2 指示】ボックス(青・赤・緑)と指定場所(緑・青・赤)の順番が異なる環境で同色の場所へPlaceせよ。",
        ),
        (
            video3_path,
            "Part 3/3",
            "【タスク3 指示】指定場所の色が異なる(2つ目の赤の置き場所が無い)場合は、置かずに元の位置へ戻せ。",
        ),
    ]

    total_written = slide_frames
    for v_path, badge_label, jp_inst in parts:
        banner_bgr, alpha_3ch = create_overlay_banner(width, badge_label, jp_inst)
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
                annotated = apply_banner(frame, banner_bgr, alpha_3ch, y_offset=68)
                writer.write(annotated)
                part_frames += 1
                total_written += 1
            frame_idx += 1
        cap.release()
        print(f"Added {v_path} at 2x speed: {part_frames} frames ({part_frames / fps:.1f}s)")

    # 3. Ending slide (PAI応用1_最終課題_スライド.jpg)
    for _ in range(slide_frames):
        writer.write(slide_bgr)
    total_written += slide_frames

    writer.release()

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
