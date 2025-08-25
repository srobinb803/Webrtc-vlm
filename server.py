#!/usr/bin/env python3
"""
WebRTC server with dual-mode support (server/WASM) for object detection.
"""
import os
import time
import json
import asyncio
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import numpy as np
import cv2
from av import VideoFrame
from statistics import median
from typing import Dict, List

# Import utility functions
from models.yolov5_utils import letterbox, process_detections, draw_detections, COCO_NAMES

# Configuration
MODE = os.environ.get("MODE", "server")  # server or wasm
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "models/yolov5n.onnx")
MODEL_IMG_SIZE = int(os.environ.get("MODEL_IMG_SIZE", "640"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))

# Application state
ROOT = os.path.join(os.path.dirname(__file__), "frontend")
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
pcs: Dict[str, RTCPeerConnection] = {}
tracks: Dict[str, MediaStreamTrack] = {}
relay = MediaRelay()
ws_clients: List[web.WebSocketResponse] = []

# Metrics state
metrics_lock = asyncio.Lock()
_inference_times_ms: List[float] = []
_processed_frames = 0
_metrics_start_time = None
MAX_METRIC_HISTORY = 20000
_frame_counter = 0

# Load ONNX model only in server mode
if MODE == "server":
    print("[server] Loading ONNX model:", MODEL_PATH)
    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    print("[server] Model input:", input_name, input_shape)
else:
    print("[server] Running in WASM mode - no server-side inference")

# COCO class names

names_map = {i: n for i, n in enumerate(COCO_NAMES)}

async def broadcast_json(obj: Dict[str, any]):
    """Broadcast JSON to all WebSocket clients."""
    text = json.dumps(obj)
    to_remove = []
    
    for ws in list(ws_clients):
        try:
            await ws.send_str(text)
        except Exception:
            to_remove.append(ws)
    
    for r in to_remove:
        try:
            ws_clients.remove(r)
        except Exception:
            pass

async def metrics_json():
    """Generate metrics JSON response."""
    async with metrics_lock:
        if not _inference_times_ms:
            return {
                "mode": MODE,
                "server_median_inference_ms": None,
                "server_p95_inference_ms": None,
                "processed_frames": _processed_frames,
                "processed_fps": 0.0,
                "duration_s": (time.time() - _metrics_start_time) if _metrics_start_time else 0.0
            }
        
        times = sorted(_inference_times_ms)
        med = float(median(times))
        p95 = float(np.percentile(times, 95.0))
        duration = (time.time() - _metrics_start_time) if _metrics_start_time else 0.0
        fps = (_processed_frames / duration) if duration > 0 else 0.0
        
        return {
            "mode": MODE,
            "server_median_inference_ms": med,
            "server_p95_inference_ms": p95,
            "processed_frames": _processed_frames,
            "processed_fps": fps,
            "duration_s": duration
        }

def create_transform_track(source: MediaStreamTrack, tag: str, client_mode: str = "server"):
    """Create a transform track that processes video based on server global MODE and per-client mode."""
    class TransformTrack(MediaStreamTrack):
        kind = "video"
        
        def __init__(self, src):
            super().__init__()
            self.src = src
            self._count = 0
            self._onnx_err_count = 0

        async def recv(self):
            global _processed_frames, _metrics_start_time, _inference_times_ms, _frame_counter
            
            frame = await self.src.recv()
            
            while True:
                try:
                    # Wait for the next frame with a tiny timeout
                    next_frame = await asyncio.wait_for(self.src.recv(), timeout=0.005)
                    # If we got one, discard the old one and keep the new one
                    frame = next_frame
                except asyncio.TimeoutError:
                    # If we timed out, it means the buffer is empty and `frame` is the latest.
                    break
                
            self._count += 1
            
            # Convert to numpy array
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            # Decide whether to do server-side processing
            do_server_processing = (MODE == "server") and (client_mode == "server")
            detections = []
            original_capture_ts = 0
            if frame.pts and frame.time_base:
                original_capture_ts = int((frame.pts * frame.time_base) * 1000)

            # 2. Check if the timestamp is valid. A timestamp from the last hour is considered valid.
            #    This prevents using initial frames where pts might be 0.
            current_time_ms = int(time.time() * 1000)
            if (current_time_ms - original_capture_ts) > 3600 * 1000:
                # 3. If NOT valid, use the server's current time as a fallback.
                capture_ts_ms = current_time_ms
            else:
                # 4. If valid, use the original timestamp.
                capture_ts_ms = original_capture_ts
            

            if do_server_processing:
                # Initialize metrics if needed
                if _metrics_start_time is None:
                    _metrics_start_time = time.time()

                # Determine model layout and preprocess (same as before)...
                # (keep your existing preprocessing/onnx inference logic here)
                try:
                    # existing preprocessing & inference block:
                    layout = "NCHW"
                    target_h = MODEL_IMG_SIZE
                    target_w = MODEL_IMG_SIZE
                    try:
                        s = input_shape
                        if isinstance(s, (list, tuple)) and len(s) == 4:
                            a, b, c, d = s
                            if isinstance(b, int) and b == 3:
                                layout = "NCHW"
                                target_h = c if isinstance(c, int) else MODEL_IMG_SIZE
                                target_w = d if isinstance(d, int) else MODEL_IMG_SIZE
                            elif isinstance(d, int) and d == 3:
                                layout = "NHWC"
                                target_h = a if isinstance(a, int) else MODEL_IMG_SIZE
                                target_w = b if isinstance(b, int) else MODEL_IMG_SIZE
                    except Exception:
                        pass

                    padded, scale, pad_x, pad_y = letterbox(img, new_shape=(target_h, target_w))
                    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

                    if layout == "NCHW":
                        inp = padded_rgb.astype(np.float32) / 255.0
                        inp = np.transpose(inp, (2, 0, 1))[None, ...]  # 1,C,H,W
                    else:
                        inp = padded_rgb.astype(np.float32) / 255.0
                        inp = np.expand_dims(inp, 0)  # 1,H,W,3

                    # Run inference
                    try:
                        t0 = time.time()
                        outputs = sess.run(None, {input_name: inp})
                        t1 = time.time()
                        inference_ms = (t1 - t0) * 1000.0
                    except Exception as e:
                        self._onnx_err_count += 1
                        outputs = None
                        inference_ms = None
                        if self._onnx_err_count < 5:
                            print("[onnx] Inference error:", e)

                    # Process detections if outputs present
                    if outputs is not None:
                        meta = {
                            'orig_w': w, 'orig_h': h, 'scale': scale,
                            'pad_x': pad_x, 'pad_y': pad_y, 'img_size': max(target_w, target_h)
                        }
                        detections = process_detections(outputs, meta, CONF_THRESHOLD, IOU_THRESHOLD)

                        # Update metrics asynchronously
                        async def update_metrics(ms):
                            global _processed_frames, _inference_times_ms
                            async with metrics_lock:
                                if ms is not None:
                                    _inference_times_ms.append(ms)
                                    if len(_inference_times_ms) > MAX_METRIC_HISTORY:
                                        _inference_times_ms = _inference_times_ms[-MAX_METRIC_HISTORY:]
                                _processed_frames += 1
                        asyncio.create_task(update_metrics(inference_ms))

                        # Broadcast detections to any websocket clients (only in server processing)
                        _frame_counter += 1
                        msg = {
                            "frame_id": str(_frame_counter),
                            "capture_ts": capture_ts_ms,
                            "recv_ts": int(t0 * 1000),
                            "inference_ts": int(t1 * 1000) if outputs is not None else None,
                            "detections": detections,
                            "mode": MODE
                        }
                        asyncio.ensure_future(broadcast_json(msg))
                except Exception as e:
                    print("[server] Exception during server processing:", e)

            # Draw overlays only when server processed for this client
            if do_server_processing and detections:
                overlay_img = draw_detections(img, detections, names_map)
            else:
                # pass-through (no overlays)
                overlay_img = img

            # Create output frame
            new_frame = VideoFrame.from_ndarray(overlay_img, format="bgr24")
            try:
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
            except Exception:
                pass
            
            return new_frame

    return TransformTrack(source)


# REST endpoint handlers (same as before)
async def offer_phone(request):
    """Handle phone offer."""
    tag = request.query.get("tag")
    if not tag:
        return web.json_response({"error": "missing_tag"}, status=400)
    
    params = await request.json()
    sdp = params.get("sdp")
    typ = params.get("type", "offer")
    
    if not sdp:
        return web.json_response({"error": "missing_sdp"}, status=400)

    pc = RTCPeerConnection()
    pcs[tag] = pc
    print(f"[pc-in] Created for tag={tag}")

    @pc.on("connectionstatechange")
    async def on_conn():
        print(f"[pc-in] {tag} state {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.pop(tag, None)
            tracks.pop(tag, None)

    @pc.on("track")
    def on_track(track):
        print(f"[pc-in] {tag} got track {track.kind}")
        if track.kind == "video":
            tracks[tag] = track

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

# Modified offer_view to accept client mode in query and create transform with that mode
async def offer_view(request):
    """Handle viewer offer."""
    tag = request.query.get("tag")
    client_mode = request.query.get("mode", "server")  # 'server' or 'wasm'
    if not tag:
        return web.json_response({"error": "missing_tag"}, status=400)
    
    params = await request.json()
    sdp = params.get("sdp")
    typ = params.get("type", "offer")
    
    if not sdp:
        return web.json_response({"error": "missing_sdp"}, status=400)

    # Wait for track to be available
    wait_until = time.time() + 15.0
    while time.time() < wait_until and tag not in tracks:
        await asyncio.sleep(0.2)
    
    if tag not in tracks:
        return web.json_response({"error": "source_not_found"}, status=404)

    source = relay.subscribe(tracks[tag])
    # Pass client_mode into transform track so server knows whether to run inference/overlay
    transformed = create_transform_track(source, tag, client_mode)

    pc = RTCPeerConnection()
    print(f"[pc-out] Created viewer pc for tag={tag} (client_mode={client_mode})")

    @pc.on("connectionstatechange")
    async def on_conn_out():
        print(f"[pc-out] {tag} state {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    pc.addTrack(transformed)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})



async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.append(ws)
    print("[ws] Client connected, total:", len(ws_clients))
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    # Handle different message types
                    if data.get("type") == "metrics_dump":
                        m = await metrics_json()
                        with open("metrics.json", "w") as f:
                            json.dump(m, f, indent=2)
                        await ws.send_str(json.dumps({"type": "metrics_stored"}))
                    
                    # Handle stream control messages
                    elif data.get("type") == "stream_stopped":
                        room = data.get("room")
                        if room in tracks:
                            # Remove the track for this room
                            tracks.pop(room, None)
                            print(f"[ws] Stream stopped for room: {room}")
                            
                            # Notify all viewers that the stream has stopped
                            await broadcast_json({
                                "type": "stream_stopped",
                                "room": room
                            })
                    
                except Exception as e:
                    print(f"[ws] Error processing message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                print("ws error", ws.exception())
    finally:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass
        print("[ws] Disconnected, total:", len(ws_clients))
    
    return ws
async def index(request):
    """Serve index page."""
    return web.FileResponse(os.path.join(ROOT, "index.html"))

async def metrics_http(request):
    """Serve metrics."""
    j = await metrics_json()
    return web.json_response(j)

# Create application
app = web.Application()
app.router.add_get("/", index)
app.router.add_post("/offer", offer_phone)
app.router.add_post("/view", offer_view)
app.router.add_get("/ws", ws_handler)
app.router.add_get("/metrics.json", metrics_http)
app.router.add_static("/", path=ROOT, show_index=True)
app.router.add_static("/models", path=MODELS_ROOT, show_index=True)

if __name__ == "__main__":
    print(f"Starting server on :8080 in {MODE} mode")
    web.run_app(app, host="0.0.0.0", port=8080)