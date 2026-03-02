#!/bin/bash
# ──────────────────────────────────────────────────────────────
# G1 Camera → GStreamer RTP/UDP stream (RGB only)
# Run this on the G1 robot.
#
# For depth streaming, also run:  python3 depth_stream.py
#
# Usage:
#   ./gst_stream.sh [DEST_IP] [PORT]
#
# Default: streams to 192.168.123.222:5600
# ──────────────────────────────────────────────────────────────

DEST_IP="${1:-192.168.123.222}"
PORT="${2:-5600}"
WIDTH="${3:-640}"
HEIGHT="${4:-480}"
FPS="${5:-15}"
DEVICE="${GST_CAMERA_DEV:-/dev/video4}"

echo "[gst_stream] Streaming ${DEVICE} → rtp://${DEST_IP}:${PORT} (${WIDTH}x${HEIGHT} @ ${FPS}fps)"
echo "[gst_stream] Press Ctrl+C to stop"

# Stop the cv container if it is holding the camera
sudo docker stop cv 2>/dev/null

gst-launch-1.0 -v \
  v4l2src device="${DEVICE}" \
  ! "video/x-raw,format=YUY2,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" \
  ! videoconvert \
  ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=15 \
  ! "video/x-h264,profile=baseline" \
  ! rtph264pay config-interval=1 pt=96 \
  ! udpsink host="${DEST_IP}" port="${PORT}" sync=false
