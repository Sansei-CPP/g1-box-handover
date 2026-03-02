"""Task – Full State Machine for G1 Robot
==========================================
Receives detection data (JSON) from camera_yolo.py via UDP.
Uses a state machine to:
  1) rotate → align → approach BOX → grab (phase 2 arms)
  2) rotate → align → approach HUMAN → hand over (phase 1 arms)

States:
  ROTATING:           Spin until box is detected
  ALIGNING:           Center box in camera frame
  APPROACHING:        Walk toward box, stop at target distance
  DONE:               Stop, switch to phase 2 arms
  ROTATING_HUMAN:     Spin until human is detected
  ALIGNING_HUMAN:     Center human in camera frame
  APPROACHING_HUMAN:  Walk toward human, stop at target distance
  FINAL:              Stop, switch back to phase 1 arms

Run camera_yolo.py first:
  python camera_yolo.py --udp-ip 127.0.0.1 --udp-port 9999

Then run this script:
  python task.py --real --network enp3s0
"""

import time
import json
import socket
import argparse
import threading
import numpy as np
import pinocchio as pin
from unitree_controller import UnitreeController
from robot_arm import G1_29_ArmController
from robot_arm_ik import G1_29_ArmIK

# ── UDP Settings ────────────────────────────────────────────────
UDP_IP = "0.0.0.0"
UDP_PORT = 9999

# ── Tunable Parameters ──────────────────────────────────────────
ROTATION_SPEED    = -1   # yaw rate while searching (rad/s)
ALIGNMENT_KP      = 0.5   # proportional gain for centering
FORWARD_SPEED     = 0.6    # base forward speed (m/s)
CENTER_THRESHOLD  = 30     # pixels from center to be "aligned"
TARGET_DISTANCE   = 0.1    # stop when this close (meters)
DIST_THRESHOLD    = 0.05   # acceptable distance error (meters)
MAX_LOST_FRAMES   = 15     # allow brief occlusions before reverting
APPROACH_TIMEOUT  = 5.0   # max seconds in APPROACHING before auto-stop
DEPTH_STALL_TIME  = 10.0    # if depth doesn't improve for this long, stop
DEPTH_STALL_DELTA = 0.03   # minimum depth change to count as progress (m)

# ── State Labels ────────────────────────────────────────────────
ROTATING          = "ROTATING"
ALIGNING          = "ALIGNING"
APPROACHING       = "APPROACHING"
DONE              = "DONE"
ROTATING_HUMAN    = "ROTATING_HUMAN"
ALIGNING_HUMAN    = "ALIGNING_HUMAN"
APPROACHING_HUMAN = "APPROACHING_HUMAN"
FINAL             = "FINAL"

# ── Arm Target Poses ───────────────────────────────────────────
# Phase 1: arms wide/forward (while searching & approaching)
L_TF_PHASE1 = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.35, +0.25, 0.25]))
R_TF_PHASE1 = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.35, -0.25, 0.25]))

# Phase 2: arms closer together (when reached the box)
L_TF_PHASE2 = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.2, +0.15, 0.2]))
R_TF_PHASE2 = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.2, -0.15, 0.2]))


def recv_detection(sock):
    """Non-blocking read of the latest UDP JSON message from vision_control."""
    latest = None
    try:
        # Drain the socket to get the freshest message
        while True:
            data, _ = sock.recvfrom(4096)
            latest = data
    except BlockingIOError:
        pass

    if latest is None:
        return None

    try:
        return json.loads(latest.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _input_listener(stop_event):
    """Background thread: waits for 'E' + Enter to signal stop."""
    while not stop_event.is_set():
        try:
            line = input()
            if line.strip().upper() == 'E':
                print("[INPUT]: Exit requested.")
                stop_event.set()
                return
        except EOFError:
            return


def solve_arm_phase(arm_ik, arm_ctrl, phase):
    """Solve IK once for the given phase and send the result to the arm controller."""
    if phase == 1:
        L_tf, R_tf = L_TF_PHASE1, R_TF_PHASE1
    else:
        L_tf, R_tf = L_TF_PHASE2, R_TF_PHASE2

    current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

    sol_q, sol_tauff = arm_ik.solve_ik(
        L_tf.homogeneous,
        R_tf.homogeneous,
        current_lr_arm_q,
        current_lr_arm_dq,
    )

    if sol_q is not None:
        arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
        print(f"[ARMS]: Phase {phase} IK solved and applied.")
    else:
        print(f"[ARMS]: Phase {phase} IK solve failed!")

    return sol_q


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--network", type=str, default="lo0")
    args = parser.parse_args()

    # Setup UDP Socket (Non-blocking)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    print(f"[UDP]: Listening for detection JSON on port {UDP_PORT}")

    input("[INPUT]: Press Enter to continue...")

    controller = UnitreeController(real=args.real, network_interface=args.network)
    controller.Init()

    # ── Initialize arm IK & controller ──────────────────────────
    print("[INFO]: Initializing arm IK solver...")
    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    arm_ctrl = G1_29_ArmController(motion_mode=True, simulation_mode=(not args.real))
    arm_ctrl.speed_gradual_max()

    # Solve IK once for phase 1 (arms wide/forward)
    print("[INFO]: Solving IK for phase 1 pose...")
    solve_arm_phase(arm_ik, arm_ctrl, phase=1)

    try:
        controller.Start()
        time.sleep(3)

        input("[INPUT]: Press Enter to start...")

        # Start background thread for 'E' exit
        stop_event = threading.Event()
        input_thread = threading.Thread(target=_input_listener, args=(stop_event,), daemon=True)
        input_thread.start()

        controller.StartMoving()
        time.sleep(1)

        print("[INFO]: Type 'E' + Enter to stop at any time.")

        # ── State Machine ───────────────────────────────────────
        state = ROTATING
        # Box detection state
        lost_count = 0
        last_cx = 0
        last_img_w = 640
        found = False
        cx = 0
        depth = None
        img_w = last_img_w

        approach_start = None     # timestamp when APPROACHING began
        best_depth = None         # best (lowest) depth seen
        best_depth_time = None    # when best_depth was last updated

        # Human detection state
        human_found = False
        human_cx = 0
        human_depth = None
        human_lost_count = 0

        human_approach_start = None
        human_best_depth = None
        human_best_depth_time = None

        done_pause_time = None    # when DONE state began (for brief pause)
        DONE_PAUSE_DURATION = 2.0 # seconds to pause in DONE before rotating for human

        print(f"[STATE]: {state} — rotating to find box...")

        while True:
            det = recv_detection(sock)

            # Only update detection state when we actually receive a message
            if det is not None:
                # Box fields
                found = det.get("detected", False)
                cx = det.get("cx", 0)
                depth = det.get("depth", None)
                img_w = det.get("img_w", last_img_w)
                last_img_w = img_w

                # Human fields
                human_found = det.get("human_detected", False)
                human_cx = det.get("human_cx", 0)
                human_depth = det.get("human_depth", None)

                # Track lost frames for box
                if found:
                    lost_count = 0
                    last_cx = cx
                else:
                    lost_count += 1

                # Track lost frames for human
                if human_found:
                    human_lost_count = 0
                else:
                    human_lost_count += 1

            # ── ROTATING ──────────────────────────────────────
            if state == ROTATING:
                controller.Move(0.0, 0.0, ROTATION_SPEED)

                if found:
                    print(f"[STATE]: Box detected at cx={cx}! → ALIGNING")
                    state = ALIGNING
                    controller.Move(0.0, 0.0, 0.0)

            # ── ALIGNING ──────────────────────────────────────
            elif state == ALIGNING:
                if lost_count > MAX_LOST_FRAMES:
                    print("[STATE]: Lost box → ROTATING")
                    state = ROTATING
                    continue

                if not found:
                    continue  # keep last command during brief occlusion

                img_center = img_w // 2
                error_x = cx - img_center

                if abs(error_x) < CENTER_THRESHOLD:
                    print(f"[STATE]: Aligned (error={error_x}px) → APPROACHING")
                    state = APPROACHING
                    controller.Move(0.0, 0.0, 0.0)
                else:
                    # Proportional rotation to center box
                    normalized = error_x / img_center
                    vyaw = -ALIGNMENT_KP * normalized
                    # Clamp
                    vyaw = max(-0.5, min(0.5, vyaw))
                    controller.Move(0.0, 0.0, vyaw)

            # ── APPROACHING ───────────────────────────────────
            elif state == APPROACHING:
                if lost_count > MAX_LOST_FRAMES:
                    print("[STATE]: Lost box during approach → ROTATING")
                    state = ROTATING
                    controller.Move(0.0, 0.0, 0.0)
                    continue

                if not found:
                    continue  # keep moving during brief occlusion

                if depth is None:
                    # No depth available — walk forward slowly
                    controller.Move(FORWARD_SPEED * 0.5, 0.0, 0.0)
                    continue

                distance_error = depth - TARGET_DISTANCE

                # Track approach start time
                now = time.time()
                if approach_start is None:
                    approach_start = now
                    best_depth = depth
                    best_depth_time = now

                # Update best depth
                if depth < (best_depth - DEPTH_STALL_DELTA):
                    best_depth = depth
                    best_depth_time = now

                # Stop conditions
                timed_out = (now - approach_start) >= APPROACH_TIMEOUT
                depth_stalled = (now - best_depth_time) >= DEPTH_STALL_TIME
                close_enough = distance_error <= DIST_THRESHOLD

                if close_enough or timed_out or depth_stalled:
                    reason = "close enough" if close_enough else ("depth stalled" if depth_stalled else "timeout")
                    print(f"[STATE]: Stopping ({reason})! depth={depth:.2f}m → DONE")
                    state = DONE
                    controller.Move(0.0, 0.0, 0.0)

                    # Solve IK once for phase 2 (closer together / grab)
                    solve_arm_phase(arm_ik, arm_ctrl, phase=2)
                    done_pause_time = time.time()
                else:
                    # Speed proportional to distance, capped
                    vx = FORWARD_SPEED * min(1.0, distance_error / 1.0)
                    vx = max(0.1, min(vx, FORWARD_SPEED))

                    # Course correction while walking
                    img_center = img_w // 2
                    error_x = cx - img_center
                    normalized = error_x / img_center
                    vyaw = -ALIGNMENT_KP * 0.3 * normalized
                    if abs(error_x) < CENTER_THRESHOLD * 0.5:
                        vyaw = 0.0
                    else:
                        vyaw = max(-0.3, min(0.3, vyaw))

                    controller.Move(vx, 0.0, vyaw)

                    # Periodic status print
                    print(f"[APPROACH]: depth={depth:.2f}m err={distance_error:.2f}m vx={vx:.2f} vyaw={vyaw:.2f}")

            # ── DONE (pause then transition to human search) ──
            elif state == DONE:
                controller.Move(0.0, 0.0, 0.0)
                if done_pause_time and (time.time() - done_pause_time) >= DONE_PAUSE_DURATION:
                    print(f"[STATE]: DONE pause complete → ROTATING_HUMAN")
                    state = ROTATING_HUMAN
                    human_lost_count = 0
                else:
                    time.sleep(0.1)

            # ═══════════════════════════════════════════════════
            # ══ HUMAN PHASE ═══════════════════════════════════
            # ═══════════════════════════════════════════════════

            # ── ROTATING_HUMAN ────────────────────────────────
            elif state == ROTATING_HUMAN:
                controller.Move(0.0, 0.0, ROTATION_SPEED)

                if human_found:
                    print(f"[STATE]: Human detected at cx={human_cx}! → ALIGNING_HUMAN")
                    state = ALIGNING_HUMAN
                    controller.Move(0.0, 0.0, 0.0)

            # ── ALIGNING_HUMAN ────────────────────────────────
            elif state == ALIGNING_HUMAN:
                if human_lost_count > MAX_LOST_FRAMES:
                    print("[STATE]: Lost human → ROTATING_HUMAN")
                    state = ROTATING_HUMAN
                    continue

                if not human_found:
                    continue  # keep last command during brief occlusion

                img_center = img_w // 2
                error_x = human_cx - img_center

                if abs(error_x) < CENTER_THRESHOLD:
                    print(f"[STATE]: Human aligned (error={error_x}px) → APPROACHING_HUMAN")
                    state = APPROACHING_HUMAN
                    controller.Move(0.0, 0.0, 0.0)
                else:
                    normalized = error_x / img_center
                    vyaw = -ALIGNMENT_KP * normalized
                    vyaw = max(-0.5, min(0.5, vyaw))
                    controller.Move(0.0, 0.0, vyaw)

            # ── APPROACHING_HUMAN ─────────────────────────────
            elif state == APPROACHING_HUMAN:
                if human_lost_count > MAX_LOST_FRAMES:
                    print("[STATE]: Lost human during approach → ROTATING_HUMAN")
                    state = ROTATING_HUMAN
                    controller.Move(0.0, 0.0, 0.0)
                    continue

                if not human_found:
                    continue  # keep moving during brief occlusion

                if human_depth is None:
                    controller.Move(FORWARD_SPEED * 0.5, 0.0, 0.0)
                    continue

                distance_error = human_depth - TARGET_DISTANCE

                now = time.time()
                if human_approach_start is None:
                    human_approach_start = now
                    human_best_depth = human_depth
                    human_best_depth_time = now

                if human_depth < (human_best_depth - DEPTH_STALL_DELTA):
                    human_best_depth = human_depth
                    human_best_depth_time = now

                timed_out = (now - human_approach_start) >= APPROACH_TIMEOUT
                depth_stalled = (now - human_best_depth_time) >= DEPTH_STALL_TIME
                close_enough = distance_error <= DIST_THRESHOLD

                if close_enough or timed_out or depth_stalled:
                    reason = "close enough" if close_enough else ("depth stalled" if depth_stalled else "timeout")
                    print(f"[STATE]: Stopping ({reason})! human_depth={human_depth:.2f}m → FINAL")
                    state = FINAL
                    controller.Move(0.0, 0.0, 0.0)

                    # Switch arms back to phase 1 (hand over)
                    solve_arm_phase(arm_ik, arm_ctrl, phase=1)
                else:
                    vx = FORWARD_SPEED * min(1.0, distance_error / 1.0)
                    vx = max(0.1, min(vx, FORWARD_SPEED))

                    img_center = img_w // 2
                    error_x = human_cx - img_center
                    normalized = error_x / img_center
                    vyaw = -ALIGNMENT_KP * 0.3 * normalized
                    if abs(error_x) < CENTER_THRESHOLD * 0.5:
                        vyaw = 0.0
                    else:
                        vyaw = max(-0.3, min(0.3, vyaw))

                    controller.Move(vx, 0.0, vyaw)
                    print(f"[APPROACH_HUMAN]: depth={human_depth:.2f}m err={distance_error:.2f}m vx={vx:.2f} vyaw={vyaw:.2f}")

            # ── FINAL ─────────────────────────────────────────
            elif state == FINAL:
                controller.Move(0.0, 0.0, 0.0)
                time.sleep(0.5)

            # Emergency stop via 'E' input
            if stop_event.is_set():
                print("[INFO]: Manual stop.")
                break

            time.sleep(0.05)  # ~20 Hz loop

        # Stop the robot
        print("[INFO]: Stopping.")
        controller.Move(0.0, 0.0, 0.0)
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO]: Interrupted.")
    finally:
        # Return arms home
        print("[INFO]: Returning arms to home position...")
        arm_ctrl.ctrl_dual_arm_go_home()

        controller.Stop()
        sock.close()
        print("[INFO]: Done.")