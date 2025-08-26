import { createSession, runInference, names_map, drawDetectionsOnCtx } from './models/yolo_wasm.js';

// ---- State + DOM  ----
const state = {
    isBroadcaster: false,
    roomId: null,
    peerConnection: null,
    socket: null,
    metrics: {
        e2eLatencies: [],
        processedFrames: 0,
        startTime: Date.now(),
        fps: 0,
        lastFpsUpdate: Date.now(),
        fpsCounter: 0
    },
    mode: 'server', // 'server' or 'wasm'
    streamActive: true,
    metricsInterval: null,
    lastDetection: null,
    abortController: null,
    wasmSession: null,
    wasmReady: false,
    wasmProcessing: false,
    wasmFrameCounter: 0,
    canvasLoopId: null
};

const E = id => document.getElementById(id);
const elements = {
    status: E('status'),
    video: E('video'),
    qrContainer: E('qr-container'),
    mobileStatus: E('mobile-status'),
    exitButton: E('exit-button'),
    metricsInfo: E('metrics-info'),
    connectionInfo: E('connection-info'),
    metricStatus: E('metric-status'),
    metricFps: E('metric-fps'),
    metricLatencyMedian: E('metric-latency-median'),
    metricLatencyP95: E('metric-latency-p95'),
    metricTotalFrames: E('metric-total-frames')
};

function setStatus(msg) { if (elements.status) elements.status.textContent = msg; }
const delay = ms => new Promise(r => setTimeout(r, ms));
function generateRoomId() { return Math.random().toString(36).slice(2, 10); }

// ---- Metrics helpers  ----
function calculateMetrics() {
    const arr = state.metrics.e2eLatencies;
    if (!arr.length) return { median_latency: 0, p95_latency: 0, processed_fps: 0, total_frames: 0, duration: 0, mode: state.mode };
    const sorted = [...arr].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || median;
    const duration = (Date.now() - state.metrics.startTime) / 1000;
    const fps = state.metrics.processedFrames / (duration || 1);
    return { median_latency: median, p95_latency: p95, processed_fps: fps, total_frames: state.metrics.processedFrames, duration, mode: state.mode };
}

function updateMetricsDisplay() {
    const m = calculateMetrics();
    if (elements.metricFps) elements.metricFps.textContent = m.processed_fps.toFixed(1);
    if (elements.metricLatencyMedian) elements.metricLatencyMedian.textContent = m.median_latency > 0 ? `${m.median_latency.toFixed(1)} ms` : '--';
    if (elements.metricLatencyP95) elements.metricLatencyP95.textContent = m.p95_latency > 0 ? `${m.p95_latency.toFixed(1)} ms` : '--';
    if (elements.metricTotalFrames) elements.metricTotalFrames.textContent = m.total_frames > 0 ? m.total_frames : '--';
}

function resetMetrics() {
    state.metrics = { e2eLatencies: [], processedFrames: 0, startTime: Date.now(), fps: 0, lastFpsUpdate: Date.now(), fpsCounter: 0 };
    updateMetricsDisplay();
}

function exportMetrics() {
    const metrics = calculateMetrics();
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(metrics, null, 2));
    const a = document.createElement('a'); a.href = dataStr; a.download = "metrics.json"; document.body.appendChild(a); a.click(); a.remove();
}

// ---- WASM init ----
const modelUrl = `${window.location.origin}/models/yolov5n.onnx`;
async function initWasm() {
    state.wasmReady = false;
    try {
        if (typeof ort === 'undefined') { setStatus('Error: ONNX Runtime not loaded (ort undefined).'); console.error('ort undefined'); return; }
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
        setStatus('Loading WASM model...');
        state.wasmSession = await createSession(modelUrl, { wasmPaths: ort.env.wasm.wasmPaths, executionProviders: ['wasm'] });
        state.wasmReady = true;
        setStatus('WASM model loaded. Ready for processing.');
        console.log('[main] wasm session ready');
    } catch (e) {
        console.error('initWasm error', e);
        setStatus('Error loading WASM model: ' + (e?.message || String(e)));
        state.wasmReady = false;
    }
}

// ---- Canvas renderer ----
function startCanvasRenderer() {
    if (state.canvasLoopId) { cancelAnimationFrame(state.canvasLoopId); state.canvasLoopId = null; }

    let canvas = document.getElementById('video-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'video-canvas';
        canvas.style.position = 'relative';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        if (elements.video && elements.video.parentNode) elements.video.parentNode.insertBefore(canvas, elements.video);
    }
    if (elements.video) elements.video.style.display = 'none';
    const ctx = canvas.getContext('2d');

    function loop() {
        if (elements.video && elements.video.videoWidth > 0 && elements.video.videoHeight > 0) {
            if (canvas.width !== elements.video.videoWidth || canvas.height !== elements.video.videoHeight) {
                canvas.width = elements.video.videoWidth; canvas.height = elements.video.videoHeight; canvas.style.display = 'block';
            }
            // draw video frame
            ctx.drawImage(elements.video, 0, 0, canvas.width, canvas.height);

            // draw latest detection overlays (WASM mode)
            if (state.mode === 'wasm' && state.lastDetection && Array.isArray(state.lastDetection.detections)) {
                drawDetectionsOnCtx(ctx, state.lastDetection.detections, names_map);
            }
        }
        state.canvasLoopId = requestAnimationFrame(loop);
    }
    state.canvasLoopId = requestAnimationFrame(loop);
}

function stopCanvasRenderer() {
    if (state.canvasLoopId) { cancelAnimationFrame(state.canvasLoopId); state.canvasLoopId = null; }
    const canvas = document.getElementById('video-canvas');
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    if (elements.video) elements.video.style.display = 'block';
}

// ---- WebSocket + detection handling ----
function handleWsDetection(data) {
    try {
        if (data.capture_ts) {
            const e2eLatency = Date.now() - data.capture_ts;
            state.metrics.e2eLatencies.push(e2eLatency);
            state.metrics.processedFrames++;
            state.metrics.fpsCounter++;
            const now = Date.now();
            if (now - state.metrics.lastFpsUpdate >= 1000) {
                state.metrics.fps = state.metrics.fpsCounter / ((now - state.metrics.lastFpsUpdate) / 1000);
                state.metrics.fpsCounter = 0;
                state.metrics.lastFpsUpdate = now;
            }
            updateMetricsDisplay();
        }
        state.lastDetection = data;
    } catch (e) { console.warn('[ws] handle detection error', e, data); }
}

function handleStreamStopped(data) {
    if (data.room === state.roomId) {
        setStatus('Phone stream stopped');
        cleanupConnections();
        showQrCode();
        resetMetrics();
        if (elements.metricStatus) elements.metricStatus.textContent = 'Disconnected';
    }
}

function connectWebSocket() {
    if (state.socket) try { state.socket.close(); } catch (e) { }
    const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`;
    try {
        const ws = new WebSocket(wsUrl);
        ws.addEventListener('open', () => console.log('[ws] connected (native)'));
        ws.addEventListener('close', () => console.log('[ws] closed (native)'));
        ws.addEventListener('error', err => console.warn('[ws] error', err));
        ws.addEventListener('message', evt => {
            try {
                const data = JSON.parse(evt.data);
                if (data.type === 'metrics_stored') console.log('[ws] metrics_stored');
                else if (data.type === 'stream_stopped') handleStreamStopped(data);
                else if (data.detections || data.capture_ts) handleWsDetection(data);
                else if (data.processed_frames !== undefined || data.server_median_inference_ms !== undefined) {
                    state.metrics.serverMetrics = data; updateMetricsDisplay();
                }
            } catch (e) { console.warn('[ws] invalid message', e, evt.data); }
        });
        state.socket = ws;
    } catch (e) { console.error('[ws] open ws failed', e); state.socket = null; }
}

// ---- WebRTC handlers ----
function waitIceGatheringComplete(pc) {
    return new Promise(resolve => {
        if (pc.iceGatheringState === 'complete') return resolve();
        function check() { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', check); resolve(); } }
        pc.addEventListener('icegatheringstatechange', check);
    });
}

function handleTrackEvent(event) {
    const stream = (event.streams && event.streams[0]) || new MediaStream([event.track]);
    if (elements.video.srcObject !== stream) {
        elements.video.srcObject = stream;
        elements.video.muted = true;
        elements.video.play().catch(() => { });
        console.log('[viewer] Attached stream');
        startCanvasRenderer();
        const playPromise = elements.video.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                state.streamActive = true;
                if (state.mode === 'wasm') {
                    // start wasm loop 
                    requestAnimationFrame(wasmProcessingLoop);
                }
            }).catch(err => { console.error('[video] Autoplay was prevented:', err); setStatus('Error: Autoplay failed. User interaction required.'); });
        }
    }
}

function handleIceConnectionStateChange() {
    const connectionState = this.iceConnectionState;
    setStatus(`Viewer connection: ${connectionState}`);
    if (elements.metricStatus) elements.metricStatus.textContent = connectionState;
    const isConnected = connectionState === 'connected' || connectionState === 'completed';
    if (isConnected) {
        showMetrics();
        if (state.mode === 'wasm') wasmProcessingLoop();
    } else {
        showQrCode();
        if (state.metricsInterval) { clearInterval(state.metricsInterval); state.metricsInterval = null; }
        if (connectionState === 'disconnected' || connectionState === 'failed') resetMetrics();
    }
}

// ---- Viewer / Broadcaster creation  ----
async function createViewer() {
    const tag = generateRoomId();
    state.roomId = tag;
    const joinUrl = `${location.origin}${location.pathname}?room=${tag}&mode=${state.mode}`;

    elements.qrContainer.innerHTML = '';
    new QRCode(elements.qrContainer, { text: joinUrl, width: 220, height: 220 });
    setStatus('Waiting for phone to join. Scan QR with phone.');

    if (state.peerConnection) { try { state.peerConnection.close(); } catch (e) { } state.peerConnection = null; }
    if (state.abortController) { try { state.abortController.abort(); } catch (e) { } state.abortController = null; }

    const pc = new RTCPeerConnection();
    state.peerConnection = pc;
    pc.ontrack = handleTrackEvent;
    pc.oniceconnectionstatechange = handleIceConnectionStateChange;
    pc.addTransceiver('video', { direction: 'recvonly' });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitIceGatheringComplete(pc);

        state.abortController = new AbortController();
        const signal = state.abortController.signal;
        const body = { sdp: pc.localDescription.sdp, type: pc.localDescription.type };

        const answer = await postWithRetry(`/view?tag=${encodeURIComponent(tag)}&mode=${encodeURIComponent(state.mode)}`, body, signal, 20, 1000);
        await pc.setRemoteDescription(answer);
        console.log('[viewer] connected for tag', tag);
        if (state.mode === 'server') connectWebSocket();
    } catch (e) {
        if (e.name === 'AbortError') console.log('[viewer] Fetch aborted');
        else { console.error('[viewer] failed', e); setStatus('Viewer error: ' + String(e)); }
    } finally {
        if (state.abortController) { state.abortController = null; }
    }
}

async function createBroadcaster(tag) {
    setStatus('Requesting camera...');
    try {
        const stream = await getCameraStream();
        elements.mobileStatus.textContent = 'Camera ready. Connecting...';
        const pc = new RTCPeerConnection();
        state.peerConnection = pc;
        pc.oniceconnectionstatechange = () => { elements.mobileStatus.textContent = `Connection: ${pc.iceConnectionState}`; };
        for (const track of stream.getTracks()) pc.addTrack(track, stream);

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitIceGatheringComplete(pc);

        const body = { sdp: pc.localDescription.sdp, type: pc.localDescription.type };
        const response = await fetch(`/offer?tag=${encodeURIComponent(tag)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const answer = await response.json();
        await pc.setRemoteDescription(answer);
        elements.mobileStatus.textContent = 'Streaming to server.';

        elements.exitButton.addEventListener('click', () => {
            pc.close();
            stream.getTracks().forEach(t => t.stop());
            elements.mobileStatus.textContent = 'Disconnected';
        });
    } catch (e) {
        console.error('[broadcaster] failed', e);
        setStatus('Broadcaster error: ' + String(e));
    }
}

// getCameraStream 
async function getCameraStream() {
    const constraints = [
        { video: { facingMode: { exact: 'environment' } }, audio: false },
        { video: { facingMode: { ideal: 'environment' } }, audio: false },
        { video: true, audio: false }
    ];
    for (const c of constraints) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(c);
            const track = stream.getVideoTracks()[0];
            const label = track?.label || '';
            if (/back|rear|environment|wide|ultrawide/i.test(label) || c === constraints[constraints.length - 1]) return stream;
            stream.getTracks().forEach(t => t.stop());
        } catch (e) { console.error('[getCameraStream] failed', e); }
    }
    return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
}

async function postWithRetry(url, body, signal, maxAttempts = 5, delayMs = 1000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal
            });

            if (!res.ok) {
                const txt = await res.text();
                if (res.status === 404 && txt.includes('source_not_found')) {
                    if (attempt < maxAttempts) {
                        await new Promise(r => setTimeout(r, delayMs));
                        continue;
                    } else {
                        throw new Error(`Max attempts reached waiting for source: ${txt}`);
                    }
                }
                throw new Error(`Request failed: ${res.status} ${txt}`);
            }
            const json = await res.json();
            return json;

        } catch (err) {
            if (err.name === 'AbortError') throw err;
            if (attempt < maxAttempts) {
                await new Promise(r => setTimeout(r, delayMs));
                continue;
            }
            throw err;
        }
    }
    throw new Error('Max retry attempts exceeded');
}


// ---- Cleanup & mode switching  ----
function cleanupConnections() {
    if (state.abortController) { try { state.abortController.abort(); } catch (e) { } state.abortController = null; }
    state.streamActive = false;
    stopCanvasRenderer();
    if (state.peerConnection) { try { state.peerConnection.close(); } catch (e) { } state.peerConnection = null; }
    if (elements.video && elements.video.srcObject) { try { elements.video.srcObject.getTracks().forEach(t => t.stop()); } catch (e) { } elements.video.srcObject = null; }
    if (state.socket) { try { if (state.socket.disconnect) state.socket.disconnect(); else state.socket.close(); } catch (e) { } state.socket = null; }
    state.lastDetection = null;
}

async function switchMode(newMode) {
    if (state.mode === newMode) return;
    state.wasmReady = false;
    cleanupConnections();
    state.mode = newMode;
    const serverModeBtn = E('server-mode-btn'), wasmModeBtn = E('wasm-mode-btn');
    if (serverModeBtn && wasmModeBtn) {
        serverModeBtn.classList.toggle('active', newMode === 'server');
        wasmModeBtn.classList.toggle('active', newMode === 'wasm');
    }
    showQrCode();
    resetMetrics();
    if (newMode === 'wasm') initWasm();
    if (state.isBroadcaster) createBroadcaster(state.roomId); else createViewer();
    setStatus(`Switched to ${newMode} mode. ${state.isBroadcaster ? 'Restarting stream...' : 'Restarting viewer...'}`);
}


function showQrCode() { if (elements.connectionInfo) elements.connectionInfo.classList.remove('hidden'); if (elements.metricsInfo) elements.metricsInfo.classList.add('hidden'); }
function showMetrics() { if (elements.connectionInfo) elements.connectionInfo.classList.add('hidden'); if (elements.metricsInfo) elements.metricsInfo.classList.remove('hidden'); }

// ---- Buttons ----
function initializeButtons() {
    const serverModeBtn = E('server-mode-btn'), wasmModeBtn = E('wasm-mode-btn'), exportBtn = E('export-metrics'), stopStreamBtn = E('stop-stream');
    if (serverModeBtn) serverModeBtn.addEventListener('click', () => switchMode('server'));
    if (wasmModeBtn) wasmModeBtn.addEventListener('click', () => switchMode('wasm'));
    if (exportBtn) exportBtn.addEventListener('click', exportMetrics);
    if (stopStreamBtn) stopStreamBtn.addEventListener('click', stopStream);
}

function stopStream() {
    const stopStreamBtn = E('stop-stream'), streamStatus = E('mobile-stream-status');
    if (stopStreamBtn && streamStatus) {
        stopStreamBtn.textContent = 'Stream Stopped';
        stopStreamBtn.disabled = true;
        streamStatus.textContent = 'Stopped';
        streamStatus.style.color = '#f44336';
        if (state.peerConnection) { state.peerConnection.close(); state.peerConnection = null; }
        if (state.socket && state.socket.readyState === WebSocket.OPEN) {
            const message = { type: 'stream_stopped', room: state.roomId };
            try { state.socket.send(JSON.stringify(message)); } catch (e) { }
        }
        if (elements.mobileStatus) elements.mobileStatus.textContent = 'Stream stopped';
        state.streamActive = false;
    }
}

// ---- WASM processing: capture -> runInference -> metrics -> store lastDetection ----
async function processResultAndUpdate(result) {
    if (!result) return;
    const { detections, processingTime } = result;
    if (!state.streamActive) return;

    // metrics
    state.metrics.e2eLatencies.push(processingTime);
    state.metrics.processedFrames++;
    state.metrics.fpsCounter++;
    updateMetricsDisplay();

    state.lastDetection = {
        frame_id: `wasm_${Date.now()}`,
        capture_ts: Date.now() - processingTime,
        inference_ts: Date.now(),
        detections: detections || [],
        mode: 'wasm'
    };
}

// captureAndInfer: capture latest video frame, call runInference, update metrics
async function captureAndInfer() {
    if (!state.streamActive || !state.wasmReady || !state.wasmSession) return;
    if (state.wasmProcessing) return;
    if (!elements.video || elements.video.videoWidth === 0 || elements.video.videoHeight === 0) return;

    state.wasmProcessing = true;
    try {
        // snap canvas sized to video native resolution (preserves orig_w/orig_h)
        const snap = document.createElement('canvas');
        snap.width = elements.video.videoWidth;
        snap.height = elements.video.videoHeight;
        const sctx = snap.getContext('2d');
        sctx.drawImage(elements.video, 0, 0, snap.width, snap.height);

        // runInference handles letterbox/preprocess internally
        const result = await runInference(state.wasmSession, snap, 0.25, 0.45);
        await processResultAndUpdate(result);
    } catch (e) {
        console.error('[wasm] captureAndInfer error', e);
    } finally {
        state.wasmProcessing = false;
    }
}

const WASM_FRAME_POLL = 120; // ms poll for stop condition; RVFC will trigger capture when available

// wasmProcessingLoop: uses RVFC if available; ensures only latest frame processed; stops cleanly
async function wasmProcessingLoop() {
    if (wasmProcessingLoop._running) return;
    wasmProcessingLoop._running = true;
    const video = elements.video;
    if (!video) { wasmProcessingLoop._running = false; return; }

    let frameCallbackHandle = null;
    let rafHandle = null;
    let lastMarker = null;
    const useRVFC = !!(video.requestVideoFrameCallback && video.cancelVideoFrameCallback);

    try {
        if (useRVFC) {
            const onFrame = (now, meta) => {
                lastMarker = meta && meta.presentedFrames !== undefined ? meta.presentedFrames : performance.now();
                if (!state.wasmProcessing) captureAndInfer().catch(e => console.error('[wasm] captureAndInfer uncaught', e));
                try { frameCallbackHandle = video.requestVideoFrameCallback(onFrame); } catch (e) { }
            };
            try { frameCallbackHandle = video.requestVideoFrameCallback(onFrame); } catch (e) { console.warn('[wasm] RVFC registration failed, falling back to RAF', e); }
        }

        if (!useRVFC || !frameCallbackHandle) {
            const rafLoop = () => {
                if (!state.streamActive || !state.wasmReady) return;
                const nowMarker = video.currentTime || performance.now();
                if (nowMarker !== lastMarker) {
                    lastMarker = nowMarker;
                    if (!state.wasmProcessing) captureAndInfer().catch(e => console.error('[wasm] captureAndInfer uncaught', e));
                }
                rafHandle = requestAnimationFrame(rafLoop);
            };
            rafHandle = requestAnimationFrame(rafLoop);
        }

        while (state.mode === 'wasm' && state.streamActive) {
            if (!video.srcObject) { state.streamActive = false; break; }
            try {
                const tracks = typeof video.srcObject.getTracks === 'function' ? video.srcObject.getTracks() : [];
                if (!tracks.length || tracks.every(t => t.readyState === 'ended')) { state.streamActive = false; break; }
            } catch (e) { console.warn('[wasm] track-check failed', e); state.streamActive = false; break; }
            await delay(WASM_FRAME_POLL);
        }

    } catch (err) { console.error('[wasm] loop error', err); }
    finally {
        try { if (useRVFC && frameCallbackHandle && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameCallbackHandle); } catch (e) { }
        try { if (rafHandle) cancelAnimationFrame(rafHandle); } catch (e) { }
        wasmProcessingLoop._running = false;
        state.wasmProcessing = false;
        console.log('[wasm] Processing loop exited cleanly.');
    }
}

async function init() {
    const urlParams = new URLSearchParams(window.location.search);
    const room = urlParams.get('room');
    const mode = urlParams.get('mode') || 'server';
    state.isBroadcaster = !!room;
    state.roomId = room;
    state.mode = mode;

    initializeButtons();
    const serverModeBtn = E('server-mode-btn'), wasmModeBtn = E('wasm-mode-btn');
    if (serverModeBtn && wasmModeBtn) {
        serverModeBtn.classList.toggle('active', state.mode === 'server');
        wasmModeBtn.classList.toggle('active', state.mode === 'wasm');
    }

    if (state.mode === 'wasm') await initWasm();

    if (state.isBroadcaster) {
        document.body.classList.add('mobile-mode');
        createBroadcaster(room);
    } else {
        document.body.classList.add('desktop-mode');
        createViewer();
    }
}

document.addEventListener('DOMContentLoaded', init);

