#!/usr/bin/env python3
"""
G1 Humanoid – Depth Streamer (TCP)
==================================
Captures depth from /dev/video0 via v4l2-ctl (Z16 format)
and streams it over TCP port 5601.

Protocol (per frame):
  Header  12 bytes:  width (u32 LE) | height (u32 LE) | data_len (u32 LE)
  Body    data_len bytes:  zlib-compressed raw depth bytes (uint16, mm)
"""

import argparse
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
import os


class DepthServer:
    def __init__(self, port=5601):
        self._port = port
        self._client = None
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", port))
        self._server.listen(1)
        self._server.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                with self._lock:
                    if self._client:
                        try:
                            self._client.close()
                        except OSError:
                            pass
                    self._client = conn
                print(f"[depth-server] Client connected: {addr}")
            except socket.timeout:
                continue

    def send(self, raw_bytes, width, height):
        with self._lock:
            if self._client is None:
                return
        compressed = zlib.compress(raw_bytes, level=1)
        header = struct.pack("<III", width, height, len(compressed))
        try:
            with self._lock:
                if self._client:
                    self._client.sendall(header + compressed)
        except (BrokenPipeError, ConnectionResetError, OSError):
            with self._lock:
                self._client = None
            print("[depth-server] Client disconnected")

    def stop(self):
        self._running = False
        with self._lock:
            if self._client:
                try:
                    self._client.close()
                except OSError:
                    pass
        self._server.close()


def main():
    parser = argparse.ArgumentParser(description="G1 Depth TCP Streamer")
    parser.add_argument("--depth-port", type=int, default=5601)
    parser.add_argument("--depth-device", default="/dev/video0")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    # Note: pixelformat must contain a trailing space 'Z16 ' for v4l2-ctl
    depth_cmd = [
        "v4l2-ctl", "-d", args.depth_device,
        f"--set-fmt-video=width={args.depth_width},height={args.depth_height},pixelformat=Z16 ",
        "--stream-mmap",
        "--stream-to=-",
    ]
    print(f"[depth] Capturing {args.depth_device} {args.depth_width}x{args.depth_height}@{args.fps}")
    
    # Run v4l2-ctl
    depth_proc = subprocess.Popen(depth_cmd, stdout=subprocess.PIPE, stderr=sys.stderr)

    server = DepthServer(port=args.depth_port)
    print(f"[depth] TCP server listening on port {args.depth_port}")

    frame_size = args.depth_width * args.depth_height * 2
    print("[stream] Running. Press Ctrl+C to stop.\n")

    frame_count = 0
    try:
        while True:
            raw = b""
            while len(raw) < frame_size:
                chunk = depth_proc.stdout.read(frame_size - len(raw))
                if not chunk:
                    print("[depth] v4l2-ctl exited unexpectedly")
                    raise KeyboardInterrupt
                raw += chunk

            server.send(raw, args.depth_width, args.depth_height)

            frame_count += 1
            if frame_count % (args.fps * 5) == 0:
                print(f"[depth] Sent {frame_count} frames")

    except KeyboardInterrupt:
        print("\n[stream] Stopping...")
    finally:
        depth_proc.terminate()
        depth_proc.wait()
        server.stop()
        print("[stream] Done.")


if __name__ == "__main__":
    main()
