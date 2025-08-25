# Real-time WebRTC VLM Multi-Object Detection

This project demonstrates a real-time, end-to-end computer vision pipeline. It streams live video from a phone's browser to a desktop browser via WebRTC, performs multi-object detection using a YOLOv5n model, and overlays the resulting bounding boxes back onto the video stream in near real-time.

The system is fully containerized with Docker for easy, one-command reproducibility and supports two distinct processing modes that can be switched dynamically from the user interface:

1. **Server Mode:** Inference is performed on a Python backend using native ONNX Runtime.
2. **WASM Mode:** Inference is performed directly in the browser using ONNX Runtime Web (WebAssembly).

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Modern web browser (Chrome recommended)
- Smartphone with modern browser

### Setup

```bash
git clone https://github.com/srobinb803/Webrtc-vlm.git
cd Webrtc-vlm
./start.sh          # Local mode
./start.sh --ngrok   # With ngrok tunnel for network issues
```

1. Open the displayed URL on your laptop
2. Scan the QR code with your phone
3. Allow camera access when prompted

## ⚙️ Using the Application

### Switching Modes

- **Server Mode:** Python backend performs inference
- **WASM Mode:** Client-side browser processing

## 📊 Benchmarking & Metrics

Click "Export Metrics" after running a stream for 30+ seconds to download performance data.

### Results (Intel i5, 8GB RAM)

**Server Mode:**

```json
{
  "median_latency_ms": 157.0,
  "p95_latency_ms": 197.0,
  "processed_fps": 3.04
}
```

**WASM Mode:**

```json
{
  "median_latency_ms": 510.9,
  "p95_latency_ms": 621.5,
  "processed_fps": 1.03
}
```

## 🔍 Troubleshooting

**Phone cannot connect:**
- Ensure your phone and laptop are on the same Wi-Fi network.
- If the problem persists, your network may be blocking peer-to-peer connections. Use the ngrok method described above.


**Docker not working properly (no video / ICE failure):**  
On some systems (especially Windows/macOS with Docker Desktop), WebRTC media may fail because Docker’s VM networking does not expose UDP ports correctly.  
If this happens, bypass Docker and run the Python server directly:

  ```bash
  python3 -m venv .venv
  source .venv/scripts/activate
  pip install -r requirements.txt
  python server.py 
  ```


## Appendix: Short Report

### Design Choices

**Architecture:** The project uses a decoupled frontend/backend architecture. The Python backend, built with aiohttp and aiortc, is responsible for WebRTC signaling, serving the frontend, and performing server-side inference. This separation allows for flexibility and scalability. The frontend is built with vanilla JavaScript, keeping it lightweight and dependency-free.

**Reproducibility:** Docker and Docker Compose for reproducible environment. The entire application and its dependencies are containerized, eliminating "it works on my machine" issues. The `start.sh` script provides a simple, user-friendly interface for starting and stopping the application.

**WASM for Low-Resource Mode:** For the low-resource requirement, ONNX Runtime Web (WASM) was chosen. It offloads all heavy computation from the server to the client. The processing loop is highly optimized, using `requestVideoFrameCallback` where available. This modern browser API syncs inference calls with the video's actual rendering pipeline for maximum efficiency.

### Backpressure Policy

To maintain low latency and prevent the video stream from falling behind, a "Process Latest" backpressure policy is implemented in both modes:

**Server Mode:** The Python aiortc track receiver is designed to aggressively drain its input buffer. When it's time to process a frame, it discards any stale, waiting frames and processes only the most recent one. This ensures the server is never wasting CPU cycles on out-of-date information.

**WASM Mode:** The `requestVideoFrameCallback` loop is paired with a "busy flag" (`state.wasmProcessing`). `requestVideoFrameCallback` ensures we only get a callback when a new, unique video frame is ready to be painted. Before running inference, the code checks the busy flag. If the previous inference task has not yet completed, the callback for the current frame is effectively skipped. This guarantees that a new frame will not be processed until the previous one is finished, preventing a backlog of inference calls and keeping the browser UI responsive.

### Analysis of Final Metrics & Trade-offs

The final metrics reveal a critical and insightful trade-off between the two modes, even though both run on the same physical CPU:

**Server Mode is High-Throughput (Lower Latency, Higher FPS):** The server's end-to-end latency is remarkably low (~157ms), and it achieves a higher processed FPS (~3.0). This is because the native Python ONNX Runtime is a highly optimized C++ backend that can leverage low-level CPU instructions. It runs as a dedicated process with minimal overhead. This mode is superior for raw performance and throughput.

**WASM Mode is Computationally Expensive (Higher Latency, Lower FPS):** The WASM mode's latency (~511ms) is significantly higher. This latency is a measure of the computation time on the client. The browser environment imposes a "sandbox tax": the WASM code cannot use the same low-level CPU optimizations, and it must compete for resources on the browser's busy main thread. While this mode has near-zero network latency, the processing itself is the bottleneck. This mode is superior for offloading server costs and for privacy, but at a performance cost on the client device.