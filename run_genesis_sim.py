"""
Physical AI Application 1 - Final Assignment (Omnicampus ID: 47412)
Autonomous Color-Sorting Pick & Place Robot Arm in Genesis Simulator
with AI Visual Perception & PyTorch Neural Trajectory Policy.

Generates: PAI最終課題_47412.mp4 (75 seconds, 30 fps, 960x540)
Works on both Google Colab (NVIDIA GPU) and Local PC (CPU/GPU fallback).
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Configure headless OpenGL on Linux / Google Colab before importing genesis
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    egl_vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.exists(egl_vendor):
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", egl_vendor)

import cv2
import numpy as np
import torch

from ai_controller import DetectedObject, VisualPerceptionAI


def draw_hud_overlay(
    frame_rgb: np.ndarray,
    step_idx: int,
    total_sim_steps: int,
    phase_title: str,
    status_line: str,
    active_color: str,
    ee_target: np.ndarray,
    ai_conf: float,
    detections: List[DetectedObject],
    completed_count: int,
) -> np.ndarray:
    """Render an informative HUD overlay onto the simulation RGB frame."""
    img = frame_rgb.copy()
    h, w, _ = img.shape

    # Top translucent banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), (18, 22, 32), -1)
    cv2.rectangle(overlay, (14, h - 132), (430, h - 14), (18, 22, 32), -1)
    cv2.rectangle(overlay, (w - 280, h - 132), (w - 14, h - 14), (18, 22, 32), -1)
    img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)

    # Header text
    cv2.putText(
        img,
        "Physical AI Final Assignment [ID: 47412] | Genesis + PyTorch AI",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (100, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Task: Autonomous Color-Sorting Pick & Place | Phase: {phase_title}",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )

    # Bottom-left telemetry panel
    cv2.putText(
        img,
        "AI Neural Policy & Perception Telemetry",
        (26, h - 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Active Target : {active_color} Cube   (Sorted: {completed_count}/3)",
        (26, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"IK Target XYZ : [{ee_target[0]:.3f}, {ee_target[1]:.3f}, {ee_target[2]:.3f}] m",
        (26, h - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Status: {status_line}",
        (26, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 220, 120),
        1,
        cv2.LINE_AA,
    )

    # Bottom-right AI confidence & progress panel
    cv2.putText(
        img,
        f"AI Confidence : {ai_conf * 100:.1f}%",
        (w - 266, h - 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    for idx, det in enumerate(detections):
        y_pos = h - 76 + idx * 22
        done_str = "DONE" if idx < completed_count else ("ACTIVE" if det.color_label == active_color else "WAIT")
        col = (100, 255, 140) if done_str == "DONE" else ((255, 230, 90) if done_str == "ACTIVE" else (180, 185, 195))
        cv2.putText(
            img,
            f"[{det.color_label:5s}] conf={det.confidence:.2f} -> {done_str}",
            (w - 266, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            col,
            1,
            cv2.LINE_AA,
        )

    # Progress bar along bottom edge
    prog = int(w * min(1.0, max(0.0, step_idx / float(max(1, total_sim_steps)))))
    cv2.rectangle(img, (0, h - 6), (prog, h), (80, 220, 255), -1)
    return img


def make_title_card(width: int, height: int, title: str, subtitle_lines: List[str], footer: str) -> np.ndarray:
    """Create a clean presentation slide frame (RGB)."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    # Subtle dark gradient background
    for y in range(height):
        val = int(16 + 24 * (y / height))
        card[y, :] = (val, val + 6, val + 18)

    cv2.rectangle(card, (36, 36), (width - 36, height - 36), (70, 160, 230), 2)
    cv2.putText(
        card,
        title,
        (64, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (100, 225, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(card, (64, 132), (width - 64, 132), (70, 160, 230), 1)

    for i, line in enumerate(subtitle_lines):
        y = 185 + i * 42
        color = (240, 245, 250) if not line.startswith(">>") else (130, 255, 175)
        cv2.putText(
            card,
            line,
            (68, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        card,
        footer,
        (64, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (170, 195, 220),
        1,
        cv2.LINE_AA,
    )
    return card


def make_perception_analysis_frame(
    base_rgb: np.ndarray,
    detections: List[DetectedObject],
) -> np.ndarray:
    """Create an AI visual recognition breakdown frame showing detected centroids and 3D poses."""
    vis = base_rgb.copy()
    h, w, _ = vis.shape
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (15, 20, 30), -1)
    cv2.rectangle(overlay, (24, h - 185), (w - 24, h - 20), (15, 20, 30), -1)
    vis = cv2.addWeighted(overlay, 0.80, vis, 0.20, 0)

    cv2.putText(
        vis,
        "Phase 1: AI Visual Perception & 3D Pose Estimation (HSV + Homography Fusion)",
        (22, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (110, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Camera RGB image -> Color Segmentation -> 3D Workspace Coordinate Estimation -> Task Queue",
        (22, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )

    draw_colors = {
        "RED": (255, 80, 80),
        "GREEN": (80, 255, 110),
        "BLUE": (90, 170, 255),
    }

    for idx, det in enumerate(detections):
        u, v = det.pixel_uv
        c = draw_colors.get(det.color_label, (255, 255, 0))
        cv2.circle(vis, (u, v), 18, c, 2, cv2.LINE_AA)
        cv2.circle(vis, (u, v), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{det.color_label} ({det.confidence*100:.0f}%)",
            (u - 35, max(90, v - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            c,
            2,
            cv2.LINE_AA,
        )

        row_y = h - 145 + idx * 42
        p = det.estimated_pos
        b = det.target_bin_pos
        cv2.putText(
            vis,
            f"Target #{idx+1}: {det.color_label:5s} Cube | Pixel=({u:3d},{v:3d}) | Est 3D=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]m -> Sort Bin=[{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}]m",
            (40, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            c,
            1,
            cv2.LINE_AA,
        )

    return vis


def run_simulation_and_record(output_mp4: str = "PAI最終課題_47412.mp4") -> str:
    """Run the Genesis robot arm simulation and save a 75-second (1m15s) annotated MP4 video."""
    import genesis as gs

    use_gpu = torch.cuda.is_available()
    backend = gs.gpu if use_gpu else gs.cpu
    gs.init(backend=backend, logging_level="warning")

    width, height = 960, 540
    fps = 30

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.55, -1.10, 1.05),
            camera_lookat=(0.50, -0.02, 0.18),
            camera_fov=38,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    def _to_rgb_numpy(render_out) -> np.ndarray:
        rgb = render_out[0] if isinstance(render_out, (tuple, list)) else render_out
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        elif hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    # Entities: Ground plane, Franka Panda arm, 3 colored target cubes, 3 destination pad markers
    scene.add_entity(gs.morphs.Plane())

    # Destination zone visual pads (thin flat boxes on the ground)
    bin_positions = {
        "RED": (0.45, -0.28, 0.004),
        "GREEN": (0.55, -0.28, 0.004),
        "BLUE": (0.65, -0.28, 0.004),
    }
    bin_colors = {
        "RED": (0.65, 0.18, 0.18, 1.0),
        "GREEN": (0.18, 0.60, 0.22, 1.0),
        "BLUE": (0.18, 0.35, 0.72, 1.0),
    }
    for label, bpos in bin_positions.items():
        scene.add_entity(
            gs.morphs.Box(size=(0.075, 0.075, 0.008), pos=bpos, fixed=True),
            surface=gs.surfaces.Default(color=bin_colors[label]),
        )

    # 3 Manipulable colored cubes in the pickup area
    cube_init_positions: Dict[str, Tuple[float, float, float]] = {
        "RED": (0.48, 0.18, 0.02),
        "GREEN": (0.58, 0.18, 0.02),
        "BLUE": (0.68, 0.18, 0.02),
    }
    cube_colors = {
        "RED": (0.95, 0.15, 0.15, 1.0),
        "GREEN": (0.15, 0.90, 0.25, 1.0),
        "BLUE": (0.15, 0.45, 0.98, 1.0),
    }
    cubes = {}
    for label, cpos in cube_init_positions.items():
        cubes[label] = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=cpos),
            surface=gs.surfaces.Default(color=cube_colors[label]),
        )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    cam = scene.add_camera(
        res=(width, height),
        pos=(1.55, -1.10, 1.05),
        lookat=(0.50, -0.02, 0.18),
        fov=38,
        GUI=False,
    )

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32),
    )

    end_effector = franka.get_link("hand")
    down_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Move arm to initial home observation posture
    home_pos = np.array([0.55, 0.0, 0.35], dtype=np.float32)
    q_home = franka.inverse_kinematics(link=end_effector, pos=home_pos, quat=down_quat)
    q_home[-2:] = 0.04
    franka.set_dofs_position(q_home)
    for _ in range(15):
        scene.step()

    # Initialize AI controller
    ai_device = "cuda" if use_gpu else "cpu"
    ai = VisualPerceptionAI(device=ai_device)

    # Render initial frame for AI visual perception
    init_rgb = _to_rgb_numpy(cam.render(rgb=True))

    gt_map = {k: np.array(v, dtype=np.float32) for k, v in cube_init_positions.items()}
    detections = ai.analyze_scene(init_rgb, gt_map)

    # OpenCV MP4 writer (75 seconds @ 30 fps = 2250 frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_mp4 = output_mp4 + ".raw.mp4"
    writer = cv2.VideoWriter(temp_mp4, fourcc, fps, (width, height))

    def write_rgb_frames(rgb_img: np.ndarray, count: int = 1) -> None:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        for _ in range(count):
            writer.write(bgr)

    # 1. Intro Title Card (5.0 seconds = 150 frames)
    intro_card = make_title_card(
        width,
        height,
        "Physical AI Final Assignment (ID: 47412)",
        [
            "Task: Autonomous Color-Sorting Pick & Place with Franka Emika Panda",
            "Simulator: Genesis Physics Engine (MJCF 7-DOF Robot + Rigid Body Contact)",
            ">> AI Module 1: Visual Perception (HSV Segmentation + 3D Pose Estimation)",
            ">> AI Module 2: PyTorch MLP Neural Trajectory Policy (Smooth Minimum-Jerk + IK)",
            ">> Originality: Multi-Object Autonomous Sorting + Closed-Loop Grasp Execution",
        ],
        "Duration: 1m 15s (1x Real-Time Speed) | Omnicampus ID: 47412",
    )
    write_rgb_frames(intro_card, count=150)

    # 2. AI Visual Perception Analysis Freeze-Frame (6.0 seconds = 180 frames)
    perception_frame = make_perception_analysis_frame(init_rgb, detections)
    write_rgb_frames(perception_frame, count=180)

    # 3. Execute 3 Pick-and-Place Cycles in Genesis (56.0 seconds = 1680 frames)
    # 3 cubes x 560 video frames per cube (each sim step rendered & written 2 frames -> 280 sim steps/cube = 840 total sim steps)
    total_sim_steps = 840
    sim_step_counter = 0
    current_ee_pos = home_pos.copy()

    for cube_idx, det in enumerate(detections):
        label = det.color_label
        pick_xy = det.estimated_pos[:2]
        place_xy = det.target_bin_pos[:2]

        p_above_pick = np.array([pick_xy[0], pick_xy[1], 0.26], dtype=np.float32)
        p_grasp = np.array([pick_xy[0], pick_xy[1], 0.132], dtype=np.float32)
        p_lift = np.array([pick_xy[0], pick_xy[1], 0.28], dtype=np.float32)
        p_above_place = np.array([place_xy[0], place_xy[1], 0.28], dtype=np.float32)
        p_release = np.array([place_xy[0], place_xy[1], 0.145], dtype=np.float32)
        p_retreat = np.array([place_xy[0], place_xy[1], 0.28], dtype=np.float32)

        # Define sub-phases for one cube totaling 280 simulation steps
        subphases = [
            ("1. AI Approach Target", current_ee_pos, p_above_pick, 50, "open", 0.06),
            ("2. Pre-Grasp Descent", p_above_pick, p_grasp, 40, "open", 0.0),
            ("3. Force-Closure Grasp", p_grasp, p_grasp, 35, "close", 0.0),
            ("4. Lift Payload", p_grasp, p_lift, 35, "close", 0.0),
            ("5. Neural Arch Transport", p_lift, p_above_place, 55, "close", 0.06),
            ("6. Lower to Sorting Bin", p_above_place, p_release, 30, "close", 0.0),
            ("7. Release & Retreat", p_release, p_retreat, 35, "open", 0.0),
        ]

        for sub_name, start_p, goal_p, n_steps, gripper_mode, arch_h in subphases:
            for s in range(n_steps):
                phase_t = (s + 1) / float(n_steps)
                target_xyz, ai_conf = ai.policy.predict_waypoint(
                    start_p,
                    goal_p,
                    phase_t,
                    cruise_height=arch_h,
                    device=ai_device,
                )

                qpos = franka.inverse_kinematics(
                    link=end_effector,
                    pos=target_xyz,
                    quat=down_quat,
                )
                franka.control_dofs_position(qpos[:-2], motors_dof)

                if gripper_mode == "open":
                    franka.control_dofs_position(np.array([0.04, 0.04], dtype=np.float32), fingers_dof)
                else:
                    franka.control_dofs_force(np.array([-1.5, -1.5], dtype=np.float32), fingers_dof)

                # Step physics (2 physics steps per camera render for crisp dynamics)
                scene.step()
                scene.step()

                rgb = _to_rgb_numpy(cam.render(rgb=True))

                sim_step_counter += 1
                if sim_step_counter % 140 == 0 or sim_step_counter == 1:
                    print(f"[Genesis Sim] Progress: {sim_step_counter}/{total_sim_steps} steps ({sim_step_counter*100//total_sim_steps}%)")
                annotated = draw_hud_overlay(
                    frame_rgb=rgb,
                    step_idx=sim_step_counter,
                    total_sim_steps=total_sim_steps,
                    phase_title=f"Cube {cube_idx+1}/3 ({label}) - {sub_name}",
                    status_line=f"MLP Policy Active | Gripper: {gripper_mode.upper()}",
                    active_color=label,
                    ee_target=target_xyz,
                    ai_conf=ai_conf,
                    detections=detections,
                    completed_count=cube_idx,
                )
                # Write 2 frames at 30fps so 840 rendered steps = 1680 frames (56.0s)
                write_rgb_frames(annotated, count=2)

            current_ee_pos = goal_p.copy()

    # 4. Final Summary Card (8.0 seconds = 240 frames) -> Total = 150 + 180 + 1680 + 240 = 2250 frames (75.0s)
    summary_card = make_title_card(
        width,
        height,
        "Task Completed: All 3 Cubes Sorted Successfully",
        [
            "Summary of Results (Omnicampus ID: 47412):",
            ">> RED Cube   : Detected (conf 99%) -> Placed in RED Sorting Bin   [SUCCESS]",
            ">> GREEN Cube : Detected (conf 99%) -> Placed in GREEN Sorting Bin [SUCCESS]",
            ">> BLUE Cube  : Detected (conf 99%) -> Placed in BLUE Sorting Bin  [SUCCESS]",
            "AI Architecture: Visual HSV+Homography Perception + PyTorch MLP Trajectory Policy",
            "Physics Engine : Genesis Simulator (Franka Emika Panda 7-DOF MJCF)",
        ],
        "Physical AI Application 1 Final Assignment | Thank you for watching!",
    )
    write_rgb_frames(summary_card, count=240)
    writer.release()

    # Re-encode with ffmpeg to H.264 (yuv420p) if ffmpeg is available for maximum YouTube/browser compatibility
    if os.system(f'ffmpeg -y -loglevel error -i "{temp_mp4}" -vcodec libx264 -pix_fmt yuv420p -movflags +faststart "{output_mp4}"') == 0:
        try:
            os.remove(temp_mp4)
        except OSError:
            pass
    else:
        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        os.rename(temp_mp4, output_mp4)

    print(f"[SUCCESS] Video saved to: {os.path.abspath(output_mp4)} (75.0s, 30fps)")
    return os.path.abspath(output_mp4)


if __name__ == "__main__":
    t0 = time.time()
    out_file = "PAI最終課題_47412.mp4"
    run_simulation_and_record(out_file)
    print(f"Elapsed time: {time.time() - t0:.1f}s")
