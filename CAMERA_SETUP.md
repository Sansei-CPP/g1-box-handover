# G1 Camera Setup Guide

## Quick Start

### 1. Start the RTSP stream on the G1

SSH into the G1 and run ffmpeg to publish the RGB camera to MediaMTX:

```bash
ssh unitree@192.168.123.164   # password: 123

# Stop the cv container (it holds /dev/video0 in a crash loop)
sudo docker stop cv

# Stream RGB camera (/dev/video4) to MediaMTX
ffmpeg -f v4l2 -input_format yuyv422 -video_size 1280x720 -framerate 15 -i /dev/video4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -rtsp_transport tcp -f rtsp "rtsp://pub:pubpass@localhost:8554/mystream"
```

### 2. Test the stream locally

```bash
# Quick verification with ffplay
ffplay -rtsp_transport tcp "rtsp://viewer:viewpass@192.168.123.164:8554/mystream"

# Or use the Python test script
python test_camera.py
```

### 3. Use in your application

```python
import cv2, os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture("rtsp://viewer:viewpass@192.168.123.164:8554/mystream", cv2.CAP_FFMPEG)

ret, frame = cap.read()  # frame is a 1280x720 BGR numpy array
```

---

## Architecture

```
G1 Robot (192.168.123.164)                    Local Machine (192.168.123.222)
┌─────────────────────────────────┐           ┌──────────────────────────┐
│                                 │           │                          │
│  /dev/video4 (RealSense RGB)    │           │  OpenCV / ffplay         │
│       │                         │           │       │                  │
│       ▼                         │           │       ▼                  │
│  ffmpeg (host)                  │   RTSP    │  cv2.VideoCapture()      │
│  - captures YUYV from v4l2     │ ────────► │  - reads H264 stream     │
│  - encodes to H264             │  TCP/8554  │  - decodes to BGR        │
│  - publishes to MediaMTX       │           │                          │
│       │                         │           └──────────────────────────┘
│       ▼                         │
│  MediaMTX (Docker container)    │
│  - RTSP server on port 8554    │
│  - publish: pub/pubpass        │
│  - read:    viewer/viewpass    │
│                                 │
└─────────────────────────────────┘
```

## RTSP Credentials

| Role    | Username | Password   | Usage                        |
|---------|----------|------------|------------------------------|
| Publish | `pub`    | `pubpass`  | ffmpeg → MediaMTX (on G1)    |
| Read    | `viewer` | `viewpass` | Client → MediaMTX (external) |

**RTSP URL (read):** `rtsp://viewer:viewpass@192.168.123.164:8554/mystream`

## Camera Devices on the G1

The G1 has an **Intel RealSense** depth camera with 6 V4L2 device nodes:

| Device       | Type  | Format | Notes                    |
|-------------|-------|--------|--------------------------|
| `/dev/video0` | Depth | Z16    | 16-bit depth map         |
| `/dev/video1` | Meta  | —      | Depth metadata           |
| `/dev/video2` | IR    | Y8I    | Infrared (greyscale)     |
| `/dev/video3` | Meta  | —      | IR metadata              |
| `/dev/video4` | **RGB** | **YUYV** | **Color camera (use this)** |
| `/dev/video5` | Meta  | —      | RGB metadata             |

RGB supported resolutions: 320×180 up to 1920×1080 (6–60 fps depending on resolution).

---

## Troubleshooting Journey

This section documents the debugging process we went through to get the camera working.

### Problem
Camera access from the local machine wasn't working. Multiple approaches failed initially.

### What We Tried (in order)

#### 1. Unitree SDK VideoClient → Error 3102
```python
from unitree_sdk2py.go2.video.video_client import VideoClient
client.GetImageSample()  # → error code 3102
```
**Root cause:** The SDK's `videohub` RPC service doesn't exist on the G1 (it's a Go2 API, not G1).

#### 2. GStreamer Multicast (230.1.1.1:1720) → No data
```bash
gst-launch-1.0 udpsrc address=230.1.1.1 port=1720 ...
```
Pipeline went to PLAYING state but **no video data arrived**. The G1 wasn't broadcasting multicast — it uses Docker-based RTSP instead.

#### 3. GStreamer via OpenCV → Not supported
```python
cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)  # fails
```
OpenCV was built **without GStreamer support** (`GStreamer: NO` in build info).

#### 4. RTSP with wrong credentials → 401 Unauthorized
```
rtsp://pub:pubpass@192.168.123.164:8554/mystream → 401
```
`pub:pubpass` are **publish** credentials. Read credentials are `viewer:viewpass`.

#### 5. RTSP with correct read creds → 404 Not Found
```
rtsp://viewer:viewpass@192.168.123.164:8554/mystream → 404
```
Auth worked, but no stream was being published.

#### 6. Discovered Docker setup
```bash
docker ps  # → mediamtx container + cv container
docker logs cv  # → crash-looping with "Invalid argument" on /dev/video0
```
The `cv` Docker container was supposed to capture from the camera and publish to MediaMTX, but it was **crash-looping** — opening `/dev/video0` (depth camera!) and immediately failing with `[Errno 22] Invalid argument`.

#### 7. Direct ffmpeg from host → /dev/video0 busy
```bash
ffmpeg -i /dev/video0 ...  # → "Device or resource busy"
```
The crash-looping `cv` container was holding the device. Fixed by `sudo docker stop cv`.

#### 8. Wrong camera device → Dark/depth images
```bash
ffmpeg -i /dev/video0 ...  # → depth stream, not RGB
```
`/dev/video0` is the **depth** sensor (Z16 format). Discovered via `v4l2-ctl --list-formats`.

#### 9. ✅ Correct device (/dev/video4) + correct format → Working!
```bash
ffmpeg -f v4l2 -input_format yuyv422 -video_size 1280x720 -framerate 15 -i /dev/video4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -rtsp_transport tcp -f rtsp "rtsp://pub:pubpass@localhost:8554/mystream"
```

### Key Lessons

1. **The G1 uses RealSense, not a simple webcam** — multiple `/dev/video*` devices, only one is RGB
2. **Camera streaming runs in Docker** — MediaMTX is a container, not a system service
3. **The original Docker setup was broken** — `cv` container crash-looped on `/dev/video0` (depth) instead of `/dev/video4` (RGB)
4. **Different credentials for publish vs read** — MediaMTX uses `pub:pubpass` for publishers and `viewer:viewpass` for readers
5. **`-pix_fmt yuv420p` is essential** — without it, ffmpeg uses `yuv444p10le` which many decoders can't handle

# G1
ffmpeg -f v4l2 -input_format yuyv422 -video_size 1280x720 -framerate 15 -i /dev/video4 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -rtsp_transport tcp -f rtsp "rtsp://pub:pubpass@localhost:8554/mystream"

# Local 
ffplay -rtsp_transport tcp "rtsp://viewer:viewpass@192.168.123.164:8554/mystream"