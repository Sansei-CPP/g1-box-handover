# python camera_yolo.py --udp-ip 127.0.0.1 --udp-port 9999
# killall -9 v4l2-ctl python3 gst-launch-1.0 2>/dev/null
"""
G1 Humanoid – GStreamer Video Client + YOLO Box & Human Detection
+ UDP Status Publisher
========================================================================
Receives H264 video (RGB) over RTP/UDP from the G1's GStreamer pipeline,
and 16-bit depth frames over TCP from realsense_stream.py.
Runs two YOLO models:
  - Box detector (custom trained best.pt)
  - Human detector (yolo26n.pt, class 0, height/width ratio >= 2)

Publishes JSON via UDP with both box and human detection data.

Usage
-----
  python camera_yolo.py --udp-ip 127.0.0.1 --udp-port 9999
"""

import argparse
import json
import time
import threading
import socket
import struct
import zlib
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

G1_IP = "192.168.123.164"


class GstCamera:
    """Receive RGB via GStreamer RTP/UDP + depth via TCP from realsense_stream.py.

    Exposes frames as numpy arrays:
      - read()       → (bool, BGR numpy)       for the RGB camera
      - read_depth() → (bool, uint16 H×W numpy) for the depth camera (16-bit, mm)

    RGB uses appsink with emit-signals. Depth uses a background thread
    connecting to the TCP depth server on the G1.
    """

    def __init__(self, port: int = 5600, depth_port: int = 5601,
                 depth_host: str = G1_IP):
        Gst.init(None)

        # ── RGB pipeline (GStreamer) ───────────────────────────────
        pipeline_str = (
            f'udpsrc port={port} '
            f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
            f'! rtph264depay '
            f'! avdec_h264 '
            f'! videoconvert '
            f'! video/x-raw,format=BGR '
            f'! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true'
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.sink.connect("new-sample", self._on_new_sample)

        self._frame = None
        self._lock = threading.Lock()

        # ── Depth receiver (TCP client) ────────────────────────────
        self._depth_host = depth_host
        self._depth_port = depth_port
        self._depth_frame = None
        self._depth_lock = threading.Lock()
        self._depth_thread = None
        self._depth_running = False

    def start(self):
        """Start the GStreamer pipeline and depth receiver."""
        # Start RGB
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

        # Start depth receiver thread
        self._depth_running = True
        self._depth_thread = threading.Thread(
            target=self._depth_receiver_loop, daemon=True
        )
        self._depth_thread.start()

        print("[camera] GStreamer RGB pipeline + depth receiver started, waiting for frames ...")

    def stop(self):
        """Stop everything."""
        self.pipeline.set_state(Gst.State.NULL)
        self._depth_running = False
        if self._depth_thread is not None:
            self._depth_thread.join(timeout=2.0)

    def read(self):
        """
        Return (success, frame) for the RGB stream.
        Pumps GLib main context to process callbacks, then returns latest frame.
        """
        ctx = GLib.MainContext.default()
        while ctx.pending():
            ctx.iteration(False)

        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def read_depth(self):
        """
        Return (success, depth_frame) for the depth stream.
        depth_frame is (H, W) uint16 numpy array (values in sensor units, ~mm).
        """
        with self._depth_lock:
            if self._depth_frame is not None:
                return True, self._depth_frame.copy()
        return False, None

    # ── RGB callback ───────────────────────────────────────────────

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct_ = caps.get_structure(0)
        w = struct_.get_value("width")
        h = struct_.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.ndarray(
            shape=(h, w, 3), dtype=np.uint8, buffer=map_info.data
        ).copy()
        buf.unmap(map_info)

        with self._lock:
            self._frame = frame

        return Gst.FlowReturn.OK

    # ── Depth receiver thread (TCP client) ─────────────────────────

    def _recvall(self, sock, n):
        """Read exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _depth_receiver_loop(self):
        """Background thread: connect to depth TCP server and receive frames."""
        while self._depth_running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                # print(f"[camera] Connecting to depth server {self._depth_host}:{self._depth_port} ...")
                sock.connect((self._depth_host, self._depth_port))
                # print(f"[camera] Connected to depth server!")
                sock.settimeout(2.0)
            except (socket.timeout, ConnectionRefusedError, OSError):
                # print(f"[camera] Depth connection failed ({e}), retrying in 2s ...")
                time.sleep(2.0)
                continue

            try:
                while self._depth_running:
                    # Protocol: [width(4)] [height(4)] [data_len(4)] [compressed_data]
                    header = self._recvall(sock, 12)
                    if header is None:
                        break

                    w, h, data_len = struct.unpack("<III", header)

                    compressed = self._recvall(sock, data_len)
                    if compressed is None:
                        break

                    try:
                        raw = zlib.decompress(compressed)
                    except zlib.error:
                        continue

                    expected = w * h * 2  # uint16
                    if len(raw) != expected:
                        continue

                    depth_arr = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))

                    with self._depth_lock:
                        self._depth_frame = depth_arr

            except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError):
                pass
            finally:
                sock.close()

            if self._depth_running:
                # print("[camera] Depth connection lost, reconnecting in 2s ...")
                time.sleep(2.0)


# ── standalone viewer ───────────────────────────────────────────

DEFAULT_MODEL = "runs/detect/runs/train/box_detector/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description="G1 GStreamer Camera Viewer + YOLO (RGB + Depth)")
    parser.add_argument("--port", type=int, default=5600,
                        help="UDP port for RGB RTP stream (default: 5600)")
    parser.add_argument("--depth-port", type=int, default=5601,
                        help="TCP port for depth stream (default: 5601)")
    parser.add_argument("--depth-host", type=str, default=G1_IP,
                        help=f"G1 IP for depth TCP connection (default: {G1_IP})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to YOLO weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--human-model", type=str, default="yolo26n.pt",
                        help="Path to YOLO weights for human detection (default: yolo26n.pt)")
    parser.add_argument("--pose-model", type=str, default="yolo26n-pose.pt",
                        help="Path to YOLO pose weights (default: yolo26n-pose.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold (default: 0.5)")
    parser.add_argument("--ratio", type=float, default=1.1,
                        help="Min height/width ratio for human detection (default: 2.0)")
    
    # NEW ARGUMENTS FOR UDP SENDING
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1",
                        help="Destination IP for detection status (default: 127.0.0.1)")
    parser.add_argument("--udp-port", type=int, default=9999,
                        help="Destination UDP port for detection status (default: 9999)")

    args = parser.parse_args()

    # ── Initialize UDP Socket ─────────────────────────────────────
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dest = (args.udp_ip, args.udp_port)
    print(f"[udp] Sending detection status to {udp_dest}")

    # ── Select device (GPU if available) ──────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[yolo] Using device: {device}")
    if device == 'cuda':
        print(f"[yolo] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load YOLO models ───────────────────────────────────────────
    print(f"[yolo] Loading box model: {args.model}")
    model = YOLO(args.model)
    model.to(device)
    print(f"[yolo] Box model loaded on {device} — classes: {model.names}")

    print(f"[yolo] Loading human model: {args.human_model}")
    human_model = YOLO(args.human_model)
    human_model.to(device)
    print(f"[yolo] Human model loaded on {device} — classes: {human_model.names}")

    print(f"[yolo] Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)
    pose_model.to(device)
    print(f"[yolo] Pose model loaded on {device}")

    # ── Start camera streams ──────────────────────────────────────
    cam = GstCamera(port=args.port, depth_port=args.depth_port,
                    depth_host=args.depth_host)
    cam.start()

    os.makedirs("recordings4", exist_ok=True)
    output_path = os.path.join("recordings4", f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

    out = None
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    print("[camera] Press 's' to save a frame, 'q' or ESC to quit.\n")

    try:
        while True:
            ret, frame = cam.read()
            ret_d, depth = cam.read_depth()

            if not ret or frame is None:
                # print("[camera] No RGB frame", end="\r")
                time.sleep(0.01)
                continue
            
            img_h, img_w = frame.shape[:2]

            # ── YOLO inference (Box) ───────────────────────────────
            results_list = model(frame, conf=args.conf, verbose=False, device=device)
            results = results_list[0]
            
            # ── Box detection ──────────────────────────────────────
            has_box = len(results.boxes) > 0

            udp_data = {"detected": False, "human_detected": False, "img_w": img_w}

            if has_box:
                # Get the first (highest confidence) detection
                box = results.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Get depth at box center (if depth available)
                box_depth = None
                if ret_d and depth is not None:
                    dh, dw = depth.shape[:2]
                    dx = int(cx * dw / img_w)
                    dy = int(cy * dh / img_h)
                    dx = max(0, min(dx, dw - 1))
                    dy = max(0, min(dy, dh - 1))
                    r = 5
                    region = depth[max(0, dy-r):min(dh, dy+r+1),
                                   max(0, dx-r):min(dw, dx+r+1)]
                    valid = region[region > 0]
                    if len(valid) > 0:
                        box_depth = float(np.median(valid)) / 1000.0  # mm → m

                udp_data["detected"] = True
                udp_data["cx"] = cx
                udp_data["cy"] = cy
                udp_data["depth"] = box_depth

            # ── YOLO inference (Human + Pose) ──────────────────────
            human_results = human_model(frame, conf=args.conf, verbose=False, classes=[0], device=device)
            pose_results = pose_model(frame, conf=args.conf, verbose=False, device=device)

            valid_humans = []
            best_kpts = None
            pose_cx, pose_cy = None, None
            human_cx, human_cy = None, None
            img_area = img_w * img_h
            for r in human_results:
                for hbox in r.boxes:
                    hx1, hy1, hx2, hy2 = hbox.xyxy[0].cpu().numpy()
                    hw = hx2 - hx1
                    hh = hy2 - hy1
                    ratio = hh / hw if hw > 0 else 0
                    box_area = hw * hh
                    area_ratio = box_area / img_area if img_area > 0 else 0
                    if ratio >= args.ratio and area_ratio >= 0.10:
                        valid_humans.append((hx1, hy1, hx2, hy2, hbox, box_area))

            if len(valid_humans) > 0:
                # Use the biggest valid human by bounding box area
                valid_humans.sort(key=lambda x: x[5], reverse=True)
                hx1, hy1, hx2, hy2, hbox, _ = valid_humans[0]
                human_cx = int((hx1 + hx2) / 2)
                human_cy = int((hy1 + hy2) / 2)

                # ── Match pose to biggest human via IoU ────────────
                best_iou = 0.0

                for pr in pose_results:
                    if not hasattr(pr, 'keypoints') or pr.keypoints is None:
                        continue
                    for pidx, pbox in enumerate(pr.boxes):
                        px1, py1, px2, py2 = pbox.xyxy[0].cpu().numpy()
                        # Compute IoU with biggest human bbox
                        xi1 = max(hx1, px1)
                        yi1 = max(hy1, py1)
                        xi2 = min(hx2, px2)
                        yi2 = min(hy2, py2)
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        area_h = (hx2 - hx1) * (hy2 - hy1)
                        area_p = (px2 - px1) * (py2 - py1)
                        union = area_h + area_p - inter
                        iou = inter / union if union > 0 else 0
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            kpts = pr.keypoints.data[pidx].cpu().numpy()  # (17, 3)
                            best_kpts = kpts

            else:
                # ── Fallback: no human bbox passed filters, use pose alone ──
                # This handles the "too close" case where the body fills the
                # frame and the bbox ratio/area filters reject it.
                best_pose_area = 0
                for pr in pose_results:
                    if not hasattr(pr, 'keypoints') or pr.keypoints is None:
                        continue
                    for pidx, pbox in enumerate(pr.boxes):
                        px1, py1, px2, py2 = pbox.xyxy[0].cpu().numpy()
                        pa = (px2 - px1) * (py2 - py1)
                        if pa > best_pose_area:
                            best_pose_area = pa
                            best_kpts = pr.keypoints.data[pidx].cpu().numpy()
                            # Use pose bbox center as initial human_cx/cy
                            human_cx = int((px1 + px2) / 2)
                            human_cy = int((py1 + py2) / 2)

            # ── Compute torso center from matched/fallback keypoints ──
            if best_kpts is not None:
                torso_indices = [5, 6, 11, 12]
                torso_pts = []
                for idx in torso_indices:
                    kx, ky, kconf = best_kpts[idx]
                    if kconf > 0.3:  # visible keypoint
                        torso_pts.append((kx, ky))
                if len(torso_pts) >= 2:
                    pose_cx = int(np.mean([p[0] for p in torso_pts]))
                    pose_cy = int(np.mean([p[1] for p in torso_pts]))

            # Override human center with pose torso center if available
            if pose_cx is not None and pose_cy is not None:
                human_cx = pose_cx
                human_cy = pose_cy

            # ── Publish human data if we have ANY detection ────────
            human_detected = (len(valid_humans) > 0) or (best_kpts is not None)
            if human_detected and human_cx is not None:
                # Get depth at human center (possibly pose-adjusted)
                human_depth = None
                if ret_d and depth is not None:
                    dh, dw = depth.shape[:2]
                    hdx = int(human_cx * dw / img_w)
                    hdy = int(human_cy * dh / img_h)
                    hdx = max(0, min(hdx, dw - 1))
                    hdy = max(0, min(hdy, dh - 1))
                    r = 5
                    region = depth[max(0, hdy-r):min(dh, hdy+r+1),
                                   max(0, hdx-r):min(dw, hdx+r+1)]
                    valid = region[region > 0]
                    if len(valid) > 0:
                        human_depth = float(np.median(valid)) / 1000.0  # mm → m

                udp_data["human_detected"] = True
                udp_data["human_cx"] = human_cx
                udp_data["human_cy"] = human_cy
                udp_data["human_depth"] = human_depth
                udp_data["pose_detected"] = (pose_cx is not None)

            # ── Send UDP ───────────────────────────────────────────
            try:
                msg_str = json.dumps(udp_data)
                udp_sock.sendto(msg_str.encode('utf-8'), udp_dest)
            except Exception as e:
                print(f"[udp] Error sending: {e}")

            # ── Annotated display ──────────────────────────────────
            annotated = results.plot()  # draw box detections

            # Draw human detections on top (cyan boxes)
            for hx1, hy1, hx2, hy2, hbox, _ in valid_humans:
                cv2.rectangle(annotated, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 255, 0), 2)
                conf_val = float(hbox.conf[0].cpu().numpy())
                cv2.putText(annotated, f"human {conf_val:.2f}", (int(hx1), int(hy1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw pose skeleton + torso center (if matched)
            if len(valid_humans) > 0 and best_kpts is not None:
                # COCO skeleton connections
                skeleton = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
                    (5, 11), (6, 12), (11, 12),                  # torso
                    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
                    (0, 1), (0, 2), (1, 3), (2, 4),              # face
                ]
                # Draw limb connections
                for i, j in skeleton:
                    kx1, ky1, kc1 = best_kpts[i]
                    kx2, ky2, kc2 = best_kpts[j]
                    if kc1 > 0.3 and kc2 > 0.3:
                        cv2.line(annotated, (int(kx1), int(ky1)), (int(kx2), int(ky2)),
                                 (0, 255, 0), 2)
                # Draw keypoints
                for kx, ky, kc in best_kpts:
                    if kc > 0.3:
                        cv2.circle(annotated, (int(kx), int(ky)), 4, (0, 255, 0), -1)
                # Draw torso center crosshair (magenta)
                if pose_cx is not None and pose_cy is not None:
                    cv2.drawMarker(annotated, (pose_cx, pose_cy), (255, 0, 255),
                                   cv2.MARKER_CROSS, 20, 3)

            # Overlay status
            status_color = (0, 255, 0) if has_box else (0, 0, 255)
            cv2.circle(annotated, (20, 20), 8, status_color, -1)
            has_pose = udp_data.get("pose_detected", False)
            status_text = f"box={has_box} human={len(valid_humans)>0} pose={has_pose}"
            if has_box and udp_data.get("depth") is not None:
                status_text += f" bd={udp_data['depth']:.2f}m"
            if len(valid_humans) > 0 and udp_data.get("human_depth") is not None:
                status_text += f" hd={udp_data['human_depth']:.2f}m"
            cv2.putText(annotated, status_text, (35, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Lazy-init video writer
            if out is None and fps > 0:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, round(fps), (w, h))
                print(f"[camera] Recording to {output_path} ({w}x{h} @ {round(fps)} fps)")

            if out is not None:
                out.write(annotated)

            # Overlay FPS counter
            display = annotated.copy()
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("G1 YOLO Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                fname = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fname, annotated)
                print(f"[camera] Saved {fname}")

    except KeyboardInterrupt:
        print("\n[camera] Interrupted.")

    finally:
        print("[camera] Done.")


if __name__ == "__main__":
    main()