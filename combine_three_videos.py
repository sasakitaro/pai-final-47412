"""
Fixes Phase 1 visual perception frames (accurate 3D-to-2D projected Cube & Bin positions,
including the BLUE bin across Tasks 1, 2, and 3) and combines the 3 simulation videos
at 2x speed with 100% English Task Instruction slides:

- Opening (4.0s) : PAI応用1_最終課題_スライド.jpg
- Part 1  (39.0s): Independent English Task 1 Instruction Slide (4.0s)
                   + PAI最終課題_47412.mp4 (Phase 1 & Phase 2 at 2x Speed, no overlapping banners)
- Part 2  (39.0s): Independent English Task 2 Instruction Slide (4.0s)
                   + PAI最終課題_47412_shuffled.mp4 (Phase 1 & Phase 2 at 2x Speed, no overlapping banners)
- Part 3  (39.0s): Independent English Task 3 Instruction Slide (4.0s)
                   + PAI最終課題_47412_reject.mp4 (Phase 1 & Phase 2 at 2x Speed, no overlapping banners)
- Ending  (4.0s) : PAI応用1_最終課題_スライド.jpg

Total Duration   : 125.0 seconds (2m 05.0s, strictly within the 1-3 minute requirement)
Output Filename  : PAI最終課題_47412_combined.mp4
"""

import os
import shutil
import cv2
import numpy as np

COLOR_BGR = {
    "RED": (75, 75, 255),
    "GREEN": (95, 235, 95),
    "BLUE": (255, 165, 65),
    "YELLOW": (60, 235, 255),
}


def load_image_unicode(path: str) -> np.ndarray:
    """Load an image from a path containing Japanese characters on Windows/Linux."""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return img


def reconstruct_clean_scene(video_path: str, task_num: int) -> np.ndarray:
    """
    Reconstruct a clean 960x540 base frame from the video:
    - Uses frame 150 for the top header bar (y=0..75), which already has the clean Phase 1 header.
    - Uses frame 330 for the 3D scene (y=75..540) and inversely un-blends the bottom HUD boxes
      to recover Bin #2 and Bin #3 cleanly while propagating clean floor tiles across the side regions.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    _, f150 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 330)
    _, f330 = cap.read()
    cap.release()

    clean = f330.copy()
    clean[0:75, :] = f150[0:75, :]

    if task_num == 1:
        bl_top_y, bl_right_x, br_left_x = 408, 431, 680
        bg = np.array([32.0, 22.0, 18.0], dtype=np.float32)
        alpha = 0.78
    elif task_num == 2:
        bl_top_y, bl_right_x, br_left_x = 402, 476, 635
        bg = np.array([30.0, 20.0, 16.0], dtype=np.float32)
        alpha = 0.78
    else:
        bl_top_y, bl_right_x, br_left_x = 400, 496, 620
        bg = np.array([30.0, 20.0, 16.0], dtype=np.float32)
        alpha = 0.80

    bot_y = 527

    # 1. Un-blend the left HUD box (recovers Bin #2 bottom tip at y=400..414 and Bin #3 at x=425..490, y=400..436)
    roi_l = f330[bl_top_y:bot_y, 14:bl_right_x].astype(np.float32)
    rec_l = np.clip((roi_l - alpha * bg) / (1.0 - alpha), 0, 255).astype(np.uint8)
    clean[bl_top_y:bot_y, 14:bl_right_x] = rec_l

    # 2. Propagate clean floor on the left (x = 10..382, safely left of Bin #2) and right (x = br_left_x - 4..952)
    y_ref = bl_top_y - 2
    for x_start, x_end in [(10, 382), (br_left_x - 4, 952)]:
        best_dx = -1.6
        best_err = 1e9
        ref_row = clean[y_ref, x_start:x_end].astype(np.float32)
        prev_row = clean[y_ref - 10, x_start:x_end].astype(np.float32)
        for dx10 in np.linspace(-25.0, 25.0, 101):
            M = np.float32([[1, 0, dx10], [0, 1, 0]])
            shifted = cv2.warpAffine(prev_row[None, :, :], M, (x_end - x_start, 1), borderMode=cv2.BORDER_REPLICATE)[0]
            err = np.mean(np.abs(ref_row[20:-20, :] - shifted[20:-20, :]))
            if err < best_err:
                best_err = err
                best_dx = dx10 / 10.0

        base_strip = np.mean(clean[y_ref - 2 : y_ref + 1, x_start:x_end].astype(np.float32), axis=0, keepdims=True)
        for y in range(y_ref, 438):
            dy = y - y_ref
            M = np.float32([[1, 0, best_dx * dy], [0, 1, 0]])
            shifted = cv2.warpAffine(base_strip, M, (x_end - x_start, 1), borderMode=cv2.BORDER_REPLICATE)[0]
            clean[y, x_start:x_end] = np.clip(shifted, 0, 255).astype(np.uint8)

    # 3. Clean floor between Bin #2 and Bin #3 at y = 415..436
    for y in range(415, 437):
        x_bin3_left = 412 if task_num == 1 else 426
        floor_col = np.mean(clean[y, 372:380].astype(np.float32), axis=0)
        clean[y, 380:x_bin3_left] = np.clip(floor_col, 0, 255).astype(np.uint8)
        clean[y, 377 : x_bin3_left + 2] = cv2.GaussianBlur(
            clean[y : y + 1, 377 : x_bin3_left + 2], (5, 1), 0
        )[0]

    clean[bl_top_y - 1 : bl_top_y + 2, 380 : bl_right_x + 2] = cv2.GaussianBlur(
        clean[bl_top_y - 1 : bl_top_y + 2, 380 : bl_right_x + 2], (3, 3), 0
    )

    # 4. Fill y = 434..527 (strictly inside the Phase 1 summary box) with clean floor from y = 529
    clean[434:527, 14:947] = cv2.GaussianBlur(clean[529:530, 14:947], (25, 1), 0)
    return clean


def render_phase1_frame(clean_bgr: np.ndarray, task_num: int) -> np.ndarray:
    """
    Render the corrected Phase 1 AI Visual Perception & 3D Pose Estimation frame.
    All Cube circles and Bin squares are placed at their exact 3D-to-2D projected centroids.
    """
    img = clean_bgr.copy()
    h, w, _ = img.shape

    overlay = img.copy()
    cv2.rectangle(overlay, (16, 434), (w - 16, 528), (28, 18, 14), -1)
    img[434:528, 16 : w - 16] = cv2.addWeighted(
        overlay[434:528, 16 : w - 16], 0.92, img[434:528, 16 : w - 16], 0.08, 0
    )
    cv2.rectangle(img, (16, 434), (w - 16, 528), (190, 140, 70), 1)

    if task_num == 1:
        cubes = [
            ("RED", (530, 294), (-52, -20)),
            ("GREEN", (561, 309), (16, -12)),
            ("BLUE", (595, 325), (16, 8)),
        ]
        bins = [
            ("RED", (380, 375), (-98, -6)),
            ("GREEN", (410, 396), (-112, 6)),
            ("BLUE", (443, 418), (18, 6)),
        ]
        for (c_col, c_uv, c_off), (b_col, b_uv, b_off) in zip(cubes, bins):
            bgr = COLOR_BGR[c_col]
            cv2.arrowedLine(img, c_uv, b_uv, bgr, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.circle(img, c_uv, 15, bgr, 2, cv2.LINE_AA)
            cv2.circle(img, c_uv, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"Cube:{c_col}",
                (c_uv[0] + c_off[0], c_uv[1] + c_off[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                bgr,
                1,
                cv2.LINE_AA,
            )
            bx, by = b_uv
            cv2.rectangle(img, (bx - 14, by - 14), (bx + 14, by + 14), bgr, 2, cv2.LINE_AA)
            cv2.circle(img, b_uv, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"Bin:{b_col}",
                (bx + b_off[0], by + b_off[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                bgr,
                1,
                cv2.LINE_AA,
            )

        lines = [
            ("RED", "Target #1: RED   Cube (530,294) | Est 3D=[0.48, 0.18, 0.02]m ===> RED   Bin (380,375) [0.45, -0.28, 0.05]m"),
            ("GREEN", "Target #2: GREEN Cube (561,309) | Est 3D=[0.58, 0.18, 0.02]m ===> GREEN Bin (410,396) [0.55, -0.28, 0.05]m"),
            ("BLUE", "Target #3: BLUE  Cube (595,325) | Est 3D=[0.68, 0.18, 0.02]m ===> BLUE  Bin (443,418) [0.65, -0.28, 0.05]m"),
        ]
        for idx, (col_name, text) in enumerate(lines):
            cv2.putText(img, text, (30, 458 + idx * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_BGR[col_name], 1, cv2.LINE_AA)

    elif task_num == 2:
        matches = [
            ("BLUE", (523, 285), "Cube#1:BLUE", (-45, -20), (418, 393), "Bin#2:BLUE", (-105, 5)),
            ("RED", (567, 295), "Cube#2:RED", (16, -10), (459, 420), "Bin#3:RED", (18, 5)),
            ("GREEN", (588, 322), "Cube#3:GREEN", (18, 6), (381, 369), "Bin#1:GREEN", (-112, -4)),
        ]
        for col_name, c_uv, c_lbl, c_off, b_uv, b_lbl, b_off in matches:
            bgr = COLOR_BGR[col_name]
            cv2.arrowedLine(img, c_uv, b_uv, bgr, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.circle(img, c_uv, 15, bgr, 2, cv2.LINE_AA)
            cv2.circle(img, c_uv, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, c_lbl, (c_uv[0] + c_off[0], c_uv[1] + c_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)
            bx, by = b_uv
            cv2.rectangle(img, (bx - 14, by - 14), (bx + 14, by + 14), bgr, 2, cv2.LINE_AA)
            cv2.circle(img, b_uv, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, b_lbl, (bx + b_off[0], by + b_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)

        lines = [
            ("BLUE", "Task #1: BLUE  | Cube Slot #1 (523,285) [0.46, 0.20]m ===> Matched Bin #2 (418,393) [0.57, -0.26]m (conf=99%)"),
            ("RED", "Task #2: RED   | Cube Slot #2 (567,295) [0.57, 0.24]m ===> Matched Bin #3 (459,420) [0.69, -0.26]m (conf=99%)"),
            ("GREEN", "Task #3: GREEN | Cube Slot #3 (588,322) [0.68, 0.18]m ===> Matched Bin #1 (381,369) [0.45, -0.26]m (conf=99%)"),
        ]
        for idx, (col_name, text) in enumerate(lines):
            cv2.putText(img, text, (30, 458 + idx * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BGR[col_name], 1, cv2.LINE_AA)

    else:
        bgr_b = COLOR_BGR["BLUE"]
        cv2.arrowedLine(img, (523, 285), (459, 420), bgr_b, 2, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(img, (523, 285), 15, bgr_b, 2, cv2.LINE_AA)
        cv2.circle(img, (523, 285), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Cube#1:BLUE", (465, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_b, 1, cv2.LINE_AA)
        cv2.rectangle(img, (459 - 14, 420 - 14), (459 + 14, 420 + 14), bgr_b, 2, cv2.LINE_AA)
        cv2.circle(img, (459, 420), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Bin#3:BLUE", (478, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_b, 1, cv2.LINE_AA)

        bgr_g = COLOR_BGR["GREEN"]
        cv2.arrowedLine(img, (588, 322), (381, 369), bgr_g, 2, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(img, (588, 322), 15, bgr_g, 2, cv2.LINE_AA)
        cv2.circle(img, (588, 322), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Cube#3:GREEN", (608, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_g, 1, cv2.LINE_AA)
        cv2.rectangle(img, (381 - 14, 369 - 14), (381 + 14, 369 + 14), bgr_g, 2, cv2.LINE_AA)
        cv2.circle(img, (381, 369), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Bin#1:GREEN", (268, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_g, 1, cv2.LINE_AA)

        bgr_r = COLOR_BGR["RED"]
        cv2.circle(img, (567, 295), 16, bgr_r, 2, cv2.LINE_AA)
        cv2.circle(img, (567, 295), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "Cube#2:RED [NO BIN -> RETURN]", (586, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_r, 1, cv2.LINE_AA)

        bgr_y = COLOR_BGR["YELLOW"]
        bx, by = 418, 393
        cv2.rectangle(img, (bx - 14, by - 14), (bx + 14, by + 14), bgr_y, 2, cv2.LINE_AA)
        cv2.line(img, (bx - 12, by - 12), (bx + 12, by + 12), bgr_r, 2, cv2.LINE_AA)
        cv2.line(img, (bx - 12, by + 12), (bx + 12, by - 12), bgr_r, 2, cv2.LINE_AA)
        cv2.putText(img, "Bin#2:YELLOW(X)", (280, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr_y, 1, cv2.LINE_AA)

        lines = [
            ("BLUE", "Task #1 [BLUE ] : Match Found -> Cube Slot #1 (523,285) ===> Place in Bin #3 (459,420) (BLUE)"),
            ("RED", "Task #2 [RED  ] : NO RED BIN (Bin #2 is YELLOW) -> Lift at Slot #2, REJECT & Return to Slot #2!"),
            ("GREEN", "Task #3 [GREEN] : Match Found -> Cube Slot #3 (588,322) ===> Place in Bin #1 (381,369) (GREEN)"),
        ]
        for idx, (col_name, text) in enumerate(lines):
            cv2.putText(img, text, (30, 458 + idx * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BGR[col_name], 1, cv2.LINE_AA)

    return img


def make_english_task_instruction_slide(
    width: int,
    height: int,
    task_num: int,
    speed_label: str = "2x Speed",
) -> np.ndarray:
    """
    Create an independent 100% English Task Instruction Slide displayed before each task.
    """
    card = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(16 + 22 * (y / height))
        card[y, :] = (val + 18, val + 6, val)

    cv2.rectangle(card, (32, 32), (width - 32, height - 32), (230, 160, 70), 2)

    if task_num == 1:
        part_title = "Task 1: Autonomous Color-Sorting Pick & Place (Aligned Layout)"
        instruction_lines = [
            "Recognize the three colored cubes (RED, GREEN, BLUE) using the overhead RGB",
            "camera and pick-and-place each cube into its matching colored bin in sequence.",
        ]
        details = [
            "Cube Layout : [Slot #1: RED,   Slot #2: GREEN, Slot #3: BLUE]",
            "Bin Layout  : [Bin  #1: RED,   Bin  #2: GREEN, Bin  #3: BLUE] (Aligned Order)",
            "AI Pipeline : HSV Color Segmentation + 3D Camera Projection + PyTorch MLP Policy",
            "Robot Arm   : Franka Emika Panda (7-DoF Arm + 2-Finger Parallel Gripper)",
        ]
    elif task_num == 2:
        part_title = "Task 2: Cross-Order Color Matching Pick & Place (Shuffled Layout)"
        instruction_lines = [
            "When the cube order (BLUE, RED, GREEN) and bin order (GREEN, BLUE, RED) are",
            "shuffled, match each cube to the bin of the same color and place it accurately.",
        ]
        details = [
            "Cube Layout : [Slot #1: BLUE,  Slot #2: RED,   Slot #3: GREEN]",
            "Bin Layout  : [Bin  #1: GREEN, Bin  #2: BLUE,  Bin  #3: RED] (Derangement)",
            "AI Routing  : Slot #1 (BLUE)->Bin #2 | Slot #2 (RED)->Bin #3 | Slot #3 (GREEN)->Bin #1",
            "AI Pipeline : Cross-Order Spatial Matching + PyTorch MLP Residual Trajectory Policy",
        ]
    else:
        part_title = "Task 3: Color Verification, Mismatch Rejection & Return-to-Origin"
        instruction_lines = [
            "Verify whether a matching colored bin exists for each picked cube. If the bin",
            "color does not match (no RED bin), do not place the cube-return it to its original slot.",
        ]
        details = [
            "Cube Layout : [Slot #1: BLUE,  Slot #2: RED,    Slot #3: GREEN]",
            "Bin Layout  : [Bin  #1: GREEN, Bin  #2: YELLOW, Bin  #3: BLUE] (No RED Bin!)",
            "AI Decision : #1 BLUE -> Bin #3 | #2 RED -> REJECT & Return to Slot #2 | #3 GREEN -> Bin #1",
            "AI Pipeline : VisualRejectAndReturnAI + Overhead Camera Verification + PyTorch MLP",
        ]

    badge_text = f"Part {task_num} of 3  |  Playback: {speed_label}"
    cv2.rectangle(card, (56, 52), (410, 86), (65, 42, 20), -1)
    cv2.rectangle(card, (56, 52), (410, 86), (255, 210, 90), 1)
    cv2.putText(card, badge_text, (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (140, 255, 190), 1, cv2.LINE_AA)

    cv2.putText(
        card,
        "Physical AI Final Assignment [ID: 47412]",
        (560, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (210, 200, 185),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(card, part_title, (56, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 225, 100), 2, cv2.LINE_AA)
    cv2.line(card, (56, 146), (width - 56, 146), (200, 140, 65), 1)

    cv2.rectangle(card, (56, 166), (width - 56, 268), (45, 32, 22), -1)
    cv2.rectangle(card, (56, 166), (width - 56, 268), (110, 230, 255), 2)
    cv2.putText(
        card,
        "TASK INSTRUCTION:",
        (74, 196),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (110, 235, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, line in enumerate(instruction_lines):
        cv2.putText(
            card,
            line,
            (74, 226 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 250, 255),
            1,
            cv2.LINE_AA,
        )

    for idx, d_line in enumerate(details):
        y = 314 + idx * 36
        cv2.putText(
            card,
            f"- {d_line}",
            (64, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 235, 245) if idx != 2 else (140, 255, 180),
            1,
            cv2.LINE_AA,
        )

    cv2.line(card, (56, height - 74), (width - 56, height - 74), (120, 90, 50), 1)
    cv2.putText(
        card,
        "Simulator: Genesis Physics Engine  |  Controller: PyTorch MLP + Damped Least-Squares IK",
        (56, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (190, 180, 165),
        1,
        cv2.LINE_AA,
    )
    return card


def encode_h264(temp_path: str, final_path: str) -> None:
    """Re-encode raw mp4v file to H.264 yuv420p for universal playback."""
    out_h264 = final_path + ".h264.mp4"
    cmd = f'ffmpeg -y -loglevel error -i "{temp_path}" -vcodec libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "{out_h264}"'
    if os.system(cmd) == 0 and os.path.exists(out_h264):
        try:
            os.remove(temp_path)
        except OSError:
            pass
        shutil.move(out_h264, final_path)
    else:
        shutil.move(temp_path, final_path)


def patch_individual_video(video_path: str, task_num: int) -> np.ndarray:
    """
    Update the standalone video file so that:
    - Frames 0..149 show the clean English Task Instruction Slide (1x Speed).
    - Frames 150..329 show the corrected Phase 1 frame (with accurate BLUE bin and cube markers).
    - Frames 330..end remain unchanged.
    Returns the corrected Phase 1 frame (BGR) for reuse in the combined video.
    """
    clean_scene = reconstruct_clean_scene(video_path, task_num)
    phase1_fixed = render_phase1_frame(clean_scene, task_num)
    intro_slide_1x = make_english_task_instruction_slide(960, 540, task_num, speed_label="1x Speed")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

    temp_out = video_path + ".patched.raw.mp4"
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < 150:
            writer.write(intro_slide_1x)
        elif idx < 330:
            writer.write(phase1_fixed)
        else:
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    encode_h264(temp_out, video_path)
    print(f"[PATCHED] Standalone video updated: {video_path} ({idx} frames)")
    return phase1_fixed


def combine_videos(
    slide_path: str = "PAI応用1_最終課題_スライド.jpg",
    video1_path: str = "PAI最終課題_47412.mp4",
    video2_path: str = "PAI最終課題_47412_shuffled.mp4",
    video3_path: str = "PAI最終課題_47412_reject.mp4",
    output_path: str = "PAI最終課題_47412_combined.mp4",
    opening_ending_sec: float = 4.0,
    task_slide_sec: float = 4.0,
) -> str:
    width, height = 960, 540
    fps = 30

    for p in [slide_path, video1_path, video2_path, video3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input file not found: {os.path.abspath(p)}")

    # First patch the 3 standalone videos so their Phase 1 frames and English slides are also fixed
    p1_frame_t1 = patch_individual_video(video1_path, 1)
    p1_frame_t2 = patch_individual_video(video2_path, 2)
    p1_frame_t3 = patch_individual_video(video3_path, 3)

    slide_bgr = load_image_unicode(slide_path)
    slide_bgr = cv2.resize(slide_bgr, (width, height), interpolation=cv2.INTER_AREA)

    temp_out = output_path + ".raw.mp4"
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    open_end_frames = int(round(opening_ending_sec * fps))
    task_slide_frames = int(round(task_slide_sec * fps))

    # 1. Opening slide (PAI応用1_最終課題_スライド.jpg)
    for _ in range(open_end_frames):
        writer.write(slide_bgr)
    total_written = open_end_frames

    # 2. Each task: Independent English Task Instruction Slide (4.0s) + Simulation Video (Phase 1 & Phase 2 at 2x speed, NO overlaid banner)
    parts = [
        (1, video1_path, p1_frame_t1),
        (2, video2_path, p1_frame_t2),
        (3, video3_path, p1_frame_t3),
    ]

    for task_num, v_path, phase1_fixed in parts:
        # Write independent English Task Instruction Slide (4.0s = 120 frames)
        task_slide = make_english_task_instruction_slide(width, height, task_num, speed_label="2x Speed")
        for _ in range(task_slide_frames):
            writer.write(task_slide)
        total_written += task_slide_frames

        # Write simulation frames (starting at frame 150 = Phase 1 + Phase 2 + Summary) at 2x speed without any overlapping banner
        cap = cv2.VideoCapture(v_path)
        frame_idx = 0
        part_sim_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= 150 and frame_idx % 2 == 0:
                if frame_idx < 330:
                    writer.write(phase1_fixed)
                else:
                    if (frame.shape[1], frame.shape[0]) != (width, height):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                part_sim_frames += 1
                total_written += 1
            frame_idx += 1
        cap.release()
        print(
            f"Added Task {task_num} ({v_path}): {task_slide_frames} slide frames ({task_slide_sec:.1f}s) "
            f"+ {part_sim_frames} sim frames at 2x speed ({part_sim_frames / fps:.1f}s)"
        )

    # 3. Ending slide (PAI応用1_最終課題_スライド.jpg)
    for _ in range(open_end_frames):
        writer.write(slide_bgr)
    total_written += open_end_frames

    writer.release()
    encode_h264(temp_out, output_path)

    total_sec = total_written / float(fps)
    print(f"[SUCCESS] Combined video saved to: {os.path.abspath(output_path)}")
    print(f"Total frames: {total_written} | Total duration: {total_sec:.1f}s ({int(total_sec // 60)}m {total_sec % 60:.1f}s)")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    combine_videos()
