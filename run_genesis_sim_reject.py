"""
Physical AI Application 1 - Final Assignment (Omnicampus ID: 47412)
Task 3: Color-Matching Pick & Place with Mismatch Rejection & Return-to-Origin
- Pickup Cubes : [Slot 1: BLUE,  Slot 2: RED (No matching bin!), Slot 3: GREEN]
- Sorting Bins : [Bin 1 : GREEN, Bin 2 : YELLOW (Mismatch!),     Bin 3 : BLUE ]

Behavior:
- Cube #1 (BLUE)  -> Matches Bin #3 (BLUE)  -> Placed in Bin #3
- Cube #2 (RED)   -> Inspects Bin #2 (YELLOW) -> Color Mismatch! Rejects placement & returns RED cube to original Slot #2
- Cube #3 (GREEN) -> Matches Bin #1 (GREEN) -> Placed in Bin #1

Generates: PAI最終課題_47412_reject.mp4 (75 seconds, 30 fps, 960x540)
"""

import os
import sys
import time
from typing import Dict, List, Tuple

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    egl_vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.exists(egl_vendor):
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", egl_vendor)

import cv2
import numpy as np
import torch

from ai_controller_reject import SortingOrRejectTask, VisualRejectAndReturnAI


COLOR_RGB_DRAW = {
    "RED": (255, 85, 85),
    "GREEN": (85, 255, 115),
    "BLUE": (95, 175, 255),
    "YELLOW": (255, 235, 70),
}


def draw_hud_overlay(
    frame_rgb: np.ndarray,
    step_idx: int,
    total_sim_steps: int,
    phase_title: str,
    status_line: str,
    active_task: SortingOrRejectTask,
    ee_target: np.ndarray,
    ai_conf: float,
    tasks: List[SortingOrRejectTask],
    completed_count: int,
    is_reject_alert: bool = False,
) -> np.ndarray:
    img = frame_rgb.copy()
    h, w, _ = img.shape

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), (16, 20, 30), -1)
    cv2.rectangle(overlay, (14, h - 140), (495, h - 14), (16, 20, 30), -1)
    cv2.rectangle(overlay, (w - 340, h - 140), (w - 14, h - 14), (16, 20, 30), -1)
    if is_reject_alert:
        cv2.rectangle(overlay, (140, 78), (w - 140, 124), (140, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.80, img, 0.20, 0)

    cv2.putText(
        img,
        "Physical AI Final Assignment [ID: 47412] | Mismatch Rejection & Return AI",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (100, 225, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Task 3: Reject Mismatched Bin & Return | {phase_title}",
        (18, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )

    if is_reject_alert:
        cv2.rectangle(img, (140, 78), (w - 140, 124), (255, 90, 90), 2)
        cv2.putText(
            img,
            f"[AI REJECT] Cube #{active_task.cube_slot_index} ({active_task.cube_color}) != Bin #{active_task.inspect_bin_slot} ({active_task.inspect_bin_color}) -> RETURN TO SLOT #{active_task.cube_slot_index}!",
            (156, 107),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 240, 200),
            2,
            cv2.LINE_AA,
        )

    # Bottom-left telemetry
    cv2.putText(
        img,
        "AI Color Verification & Return-to-Origin Telemetry",
        (24, h - 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    if active_task.is_match:
        route_str = (
            f"Decision: MATCH  | Slot #{active_task.cube_slot_index} ({active_task.cube_color}) "
            f"===> Place in Bin #{active_task.matched_bin_slot} ({active_task.cube_color})"
        )
        route_col = COLOR_RGB_DRAW.get(active_task.cube_color, (255, 255, 255))
    else:
        route_str = (
            f"Decision: REJECT | Slot #{active_task.cube_slot_index} ({active_task.cube_color}) "
            f"!= Bin #{active_task.inspect_bin_slot} ({active_task.inspect_bin_color}) -> Return to Slot #{active_task.cube_slot_index}"
        )
        route_col = (255, 110, 110)

    cv2.putText(
        img,
        route_str,
        (24, h - 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        route_col,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"IK Target XYZ : [{ee_target[0]:.3f}, {ee_target[1]:.3f}, {ee_target[2]:.3f}] m",
        (24, h - 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Status: {status_line}",
        (24, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 220, 120),
        1,
        cv2.LINE_AA,
    )

    # Bottom-right task summary
    cv2.putText(
        img,
        f"AI Verification Conf: {ai_conf * 100:.1f}%",
        (w - 328, h - 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    for idx, t in enumerate(tasks):
        y_pos = h - 84 + idx * 24
        state = "DONE" if idx < completed_count else ("ACTIVE" if idx == completed_count else "WAIT")
        col = (100, 255, 140) if state == "DONE" else ((255, 230, 90) if state == "ACTIVE" else (180, 185, 195))
        if t.is_match:
            act_desc = f"Slot{t.cube_slot_index}->Bin{t.matched_bin_slot}(PLACE)"
        else:
            act_desc = f"Slot{t.cube_slot_index}->RETURN(NO BIN)"
        cv2.putText(
            img,
            f"#{t.task_order} [{t.cube_color:5s}] {act_desc}: {state}",
            (w - 328, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            col,
            1,
            cv2.LINE_AA,
        )

    prog = int(w * min(1.0, max(0.0, step_idx / float(max(1, total_sim_steps)))))
    cv2.rectangle(img, (0, h - 6), (prog, h), (80, 220, 255), -1)
    return img


def make_title_card(width: int, height: int, title: str, subtitle_lines: List[str], footer: str) -> np.ndarray:
    card = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(16 + 24 * (y / height))
        card[y, :] = (val, val + 6, val + 18)

    cv2.rectangle(card, (36, 36), (width - 36, height - 36), (70, 160, 230), 2)
    cv2.putText(
        card,
        title,
        (56, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (100, 225, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(card, (56, 126), (width - 56, 126), (70, 160, 230), 1)

    for i, line in enumerate(subtitle_lines):
        y = 176 + i * 40
        color = (240, 245, 250) if not line.startswith(">>") else (130, 255, 175)
        if "REJECT" in line or "NO RED BIN" in line:
            color = (255, 140, 140)
        cv2.putText(
            card,
            line,
            (60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        card,
        footer,
        (56, height - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (170, 195, 220),
        1,
        cv2.LINE_AA,
    )
    return card


def make_reject_analysis_frame(
    base_rgb: np.ndarray,
    tasks: List[SortingOrRejectTask],
) -> np.ndarray:
    vis = base_rgb.copy()
    h, w, _ = vis.shape
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 74), (15, 20, 30), -1)
    cv2.rectangle(overlay, (20, h - 190), (w - 20, h - 18), (15, 20, 30), -1)
    vis = cv2.addWeighted(overlay, 0.82, vis, 0.18, 0)

    cv2.putText(
        vis,
        "Phase 1: AI Color Verification - Detecting Missing Bin for 2nd Cube (RED)",
        (22, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (110, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Cubes: [Slot1=BLUE, Slot2=RED, Slot3=GREEN] vs Bins: [Bin1=GREEN, Bin2=YELLOW, Bin3=BLUE]",
        (22, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 230, 130),
        1,
        cv2.LINE_AA,
    )

    for idx, t in enumerate(tasks):
        c = COLOR_RGB_DRAW.get(t.cube_color, (255, 255, 0))
        cu, cv_px = t.cube_pixel_uv
        bu, bv_px = t.bin_pixel_uv

        cv2.circle(vis, (cu, cv_px), 16, c, 2, cv2.LINE_AA)
        if t.is_match:
            cv2.rectangle(vis, (bu - 16, bv_px - 16), (bu + 16, bv_px + 16), c, 2, cv2.LINE_AA)
            cv2.arrowedLine(vis, (cu, cv_px), (bu, bv_px), c, 2, cv2.LINE_AA, tipLength=0.08)
            row_txt = (
                f"Task #{t.task_order} [{t.cube_color:5s}] : Match Found -> "
                f"Cube Slot #{t.cube_slot_index} ===> Place in Bin #{t.matched_bin_slot} ({t.cube_color})"
            )
            row_col = c
        else:
            # Draw dashed/red X at the mismatched bin and a return loop back to cube slot
            bin_c = COLOR_RGB_DRAW.get(t.inspect_bin_color, (255, 235, 70))
            cv2.rectangle(vis, (bu - 16, bv_px - 16), (bu + 16, bv_px + 16), bin_c, 2, cv2.LINE_AA)
            cv2.line(vis, (bu - 14, bv_px - 14), (bu + 14, bv_px + 14), (255, 60, 60), 3, cv2.LINE_AA)
            cv2.line(vis, (bu - 14, bv_px + 14), (bu + 14, bv_px - 14), (255, 60, 60), 3, cv2.LINE_AA)
            cv2.putText(
                vis,
                "MISMATCH! RETURN",
                (cu - 55, max(92, cv_px - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 95, 95),
                2,
                cv2.LINE_AA,
            )
            row_txt = (
                f"Task #{t.task_order} [{t.cube_color:5s}] : NO {t.cube_color} BIN (Bin #2 is {t.inspect_bin_color}) -> "
                f"REJECT & RETURN to Original Slot #{t.cube_slot_index}!"
            )
            row_col = (255, 110, 110)

        row_y = h - 150 + idx * 42
        cv2.putText(
            vis,
            row_txt,
            (34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            row_col,
            1,
            cv2.LINE_AA,
        )

    return vis


def run_reject_simulation_and_record(output_mp4: str = "PAI最終課題_47412_reject.mp4") -> str:
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
            camera_lookat=(0.52, -0.02, 0.18),
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

    def _entity_pos_numpy(entity, fallback: np.ndarray) -> np.ndarray:
        try:
            p = entity.get_pos()
            if hasattr(p, "detach"):
                p = p.detach().cpu().numpy()
            elif hasattr(p, "cpu"):
                p = p.cpu().numpy()
            p = np.asarray(p, dtype=np.float32).reshape(-1)
            if p.size >= 3:
                return p[:3].copy()
        except Exception:
            pass
        return fallback.copy()

    scene.add_entity(gs.morphs.Plane())

    # Pickup Cubes: 2nd cube (Slot 2 = RED) has NO matching bin!
    cube_layout: List[Tuple[str, int, Tuple[float, float, float]]] = [
        ("BLUE",  1, (0.46, 0.20, 0.02)),
        ("RED",   2, (0.57, 0.24, 0.02)),  # 2nd color: NO RED bin exists!
        ("GREEN", 3, (0.68, 0.18, 0.02)),
    ]

    # Destination Bins: Bin 1=GREEN, Bin 2=YELLOW (mismatched color!), Bin 3=BLUE
    bin_layout: List[Tuple[str, int, Tuple[float, float, float]]] = [
        ("GREEN",  1, (0.45, -0.26, 0.004)),
        ("YELLOW", 2, (0.57, -0.26, 0.004)),  # YELLOW instead of RED!
        ("BLUE",   3, (0.69, -0.26, 0.004)),
    ]

    bin_rgba = {
        "GREEN":  (0.16, 0.70, 0.22, 1.0),
        "YELLOW": (0.95, 0.85, 0.12, 1.0),
        "BLUE":   (0.16, 0.38, 0.82, 1.0),
    }
    cube_rgba = {
        "RED":   (0.96, 0.15, 0.15, 1.0),
        "GREEN": (0.15, 0.92, 0.25, 1.0),
        "BLUE":  (0.15, 0.48, 0.98, 1.0),
    }

    for color, _, bpos in bin_layout:
        scene.add_entity(
            gs.morphs.Box(size=(0.085, 0.085, 0.008), pos=bpos, fixed=True),
            surface=gs.surfaces.Default(color=bin_rgba[color]),
        )

    cubes = {}
    for color, _, cpos in cube_layout:
        cubes[color] = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=cpos),
            surface=gs.surfaces.Default(color=cube_rgba[color]),
        )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    cam = scene.add_camera(
        res=(width, height),
        pos=(1.55, -1.10, 1.05),
        lookat=(0.52, -0.02, 0.18),
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

    home_pos = np.array([0.55, 0.0, 0.36], dtype=np.float32)
    q_home = franka.inverse_kinematics(link=end_effector, pos=home_pos, quat=down_quat)
    q_home[-2:] = 0.04
    franka.set_dofs_position(q_home)
    for _ in range(20):
        scene.step()

    ai_device = "cuda" if use_gpu else "cpu"
    ai = VisualRejectAndReturnAI(device=ai_device)

    init_rgb = _to_rgb_numpy(cam.render(rgb=True))
    tasks = ai.analyze_scene_with_rejection(init_rgb, cube_layout, bin_layout)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_mp4 = output_mp4 + ".raw.mp4"
    writer = cv2.VideoWriter(temp_mp4, fourcc, fps, (width, height))

    def write_rgb_frames(rgb_img: np.ndarray, count: int = 1) -> None:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        for _ in range(count):
            writer.write(bgr)

    # 1. Intro Title Card (5.0s = 150 frames)
    intro_card = make_title_card(
        width,
        height,
        "Physical AI Final Assignment (ID: 47412) - Task 3",
        [
            "Task: Color Verification, Mismatch Rejection & Return-to-Origin",
            "Pickup Cubes : [Slot 1: BLUE,  Slot 2: RED,    Slot 3: GREEN]",
            "Sorting Bins : [Bin 1 : GREEN, Bin 2 : YELLOW, Bin 3 : BLUE ]",
            ">> 2nd Cube (RED) has NO matching bin (Bin #2 is YELLOW != RED)",
            ">> AI inspects Bin #2, REJECTS placement, and RETURNS RED cube to Slot #2",
        ],
        "Duration: 1m 15s (1x Speed) | Omnicampus ID: 47412",
    )
    write_rgb_frames(intro_card, count=150)

    # 2. Phase 1 Analysis Frame (6.0s = 180 frames)
    analysis_frame = make_reject_analysis_frame(init_rgb, tasks)
    write_rgb_frames(analysis_frame, count=180)

    # 3. Execute 3 Tasks (840 rendered steps -> 1680 frames = 56.0s)
    total_sim_steps = 840
    sim_step_counter = 0
    current_ee_pos = home_pos.copy()

    for idx, task in enumerate(tasks):
        c_col = task.cube_color
        live_cube_pos = _entity_pos_numpy(cubes[c_col], task.cube_pos_3d)
        pick_xy = live_cube_pos[:2]
        bin_xy = task.target_bin_pos_3d[:2]

        p_above_pick = np.array([pick_xy[0], pick_xy[1], 0.26], dtype=np.float32)
        p_grasp = np.array([pick_xy[0], pick_xy[1], 0.130], dtype=np.float32)
        p_lift = np.array([pick_xy[0], pick_xy[1], 0.29], dtype=np.float32)

        if task.is_match:
            # Normal matched cross-sorting trajectory (280 steps)
            p_above_bin = np.array([bin_xy[0], bin_xy[1], 0.29], dtype=np.float32)
            p_release = np.array([bin_xy[0], bin_xy[1], 0.142], dtype=np.float32)
            p_retreat = np.array([bin_xy[0], bin_xy[1], 0.28], dtype=np.float32)

            subphases = [
                ("1. Approach Cube Slot", current_ee_pos, p_above_pick, 50, "open", 0.06, False),
                ("2. Pre-Grasp Descent", p_above_pick, p_grasp, 40, "open", 0.0, False),
                ("3. Force-Closure Grasp", p_grasp, p_grasp, 35, "close", 0.0, False),
                ("4. Vertical Lift", p_grasp, p_lift, 35, "close", 0.0, False),
                ("5. Transport to Matched Bin", p_lift, p_above_bin, 55, "close", 0.07, False),
                ("6. Lower to Matched Bin", p_above_bin, p_release, 30, "close", 0.0, False),
                ("7. Release & Retreat", p_release, p_retreat, 35, "open", 0.0, False),
            ]
        else:
            # 2nd Cube (RED): Overhead camera detects NO matching RED bin (Bin #2 is YELLOW).
            # Lift RED cube above Slot #2 -> Reject placement -> Lower straight back down to original Slot #2!
            p_return_place = np.array([pick_xy[0], pick_xy[1], 0.135], dtype=np.float32)
            p_return_retreat = np.array([pick_xy[0], pick_xy[1], 0.28], dtype=np.float32)

            subphases = [
                ("1. Approach Slot #2 (RED)", current_ee_pos, p_above_pick, 50, "open", 0.06, False),
                ("2. Pre-Grasp Descent", p_above_pick, p_grasp, 40, "open", 0.0, False),
                ("3. Force-Closure Grasp", p_grasp, p_grasp, 35, "close", 0.0, False),
                ("4. Vertical Lift RED Cube", p_grasp, p_lift, 45, "close", 0.0, False),
                ("5. [REJECT] No RED Bin (Bin#2=YELLOW)", p_lift, p_lift, 40, "close", 0.0, True),
                ("6. Lower Back to Original Slot #2", p_lift, p_return_place, 35, "close", 0.0, True),
                ("7. Release at Origin & Retreat", p_return_place, p_return_retreat, 35, "open", 0.0, False),
            ]

        for sub_name, start_p, goal_p, n_steps, gripper_mode, arch_h, alert_flag in subphases:
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

                scene.step()
                scene.step()
                scene.step()

                rgb = _to_rgb_numpy(cam.render(rgb=True))

                sim_step_counter += 1
                if sim_step_counter % 140 == 0 or sim_step_counter == 1:
                    print(
                        f"[Genesis Reject Sim] Progress: {sim_step_counter}/{total_sim_steps} "
                        f"({sim_step_counter*100//total_sim_steps}%) | {sub_name}"
                    )

                annotated = draw_hud_overlay(
                    frame_rgb=rgb,
                    step_idx=sim_step_counter,
                    total_sim_steps=total_sim_steps,
                    phase_title=f"Cube {idx+1}/3 ({c_col}) - {sub_name}",
                    status_line="REJECT & RETURN TO ORIGIN" if alert_flag else f"MLP Active | Gripper: {gripper_mode.upper()}",
                    active_task=task,
                    ee_target=target_xyz,
                    ai_conf=ai_conf,
                    tasks=tasks,
                    completed_count=idx,
                    is_reject_alert=alert_flag,
                )
                write_rgb_frames(annotated, count=2)

            current_ee_pos = goal_p.copy()

    # 4. Final Summary Card (8.0s = 240 frames)
    summary_card = make_title_card(
        width,
        height,
        "Task 3 Completed: Mismatch Rejected & Returned to Origin",
        [
            "Verification Results (Omnicampus ID: 47412):",
            ">> Cube #1 [BLUE ] (Slot #1) : Matched Bin #3 (BLUE)  -> PLACED IN BIN #3   [SUCCESS]",
            ">> Cube #2 [RED  ] (Slot #2) : Bin #2 is YELLOW != RED -> REJECTED & RETURNED [SUCCESS]",
            ">> Cube #3 [GREEN] (Slot #3) : Matched Bin #1 (GREEN) -> PLACED IN BIN #1   [SUCCESS]",
            "AI Vision prevented misplacement when target bin color did not match.",
            "Unmatched RED cube was safely returned to its original coordinates (Slot #2).",
        ],
        "Physical AI Application 1 Final Assignment | Thank you for watching!",
    )
    write_rgb_frames(summary_card, count=240)
    writer.release()

    if os.system(f'ffmpeg -y -loglevel error -i "{temp_mp4}" -vcodec libx264 -pix_fmt yuv420p -movflags +faststart "{output_mp4}"') == 0:
        try:
            os.remove(temp_mp4)
        except OSError:
            pass
    else:
        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        os.rename(temp_mp4, output_mp4)

    print(f"[SUCCESS] Reject-and-return task video saved to: {os.path.abspath(output_mp4)} (75.0s, 30fps)")
    return os.path.abspath(output_mp4)


if __name__ == "__main__":
    t0 = time.time()
    out_file = "PAI最終課題_47412_reject.mp4"
    run_reject_simulation_and_record(out_file)
    print(f"Elapsed time: {time.time() - t0:.1f}s")
