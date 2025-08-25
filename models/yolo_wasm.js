export const WASM_DEFAULT_INPUT_SIZE = 640;

export const COCO_NAMES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
  "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
  "scissors","teddy bear","hair drier","toothbrush"
];
export const names_map = {};
COCO_NAMES.forEach((n, i) => { names_map[i] = n; });

export async function createSession(modelUrl, options = {}) {
  if (typeof ort === 'undefined') throw new Error('onnxruntime-web (ort) not found in global scope');

  // allow caller to override wasmPaths (same pattern as earlier)
  if (options.wasmPaths) {
    try { ort.env.wasm.wasmPaths = options.wasmPaths; } catch (e) { console.warn('failed to set wasmPaths', e); }
  }

  const executionProviders = options.executionProviders || ['wasm'];
  console.log('[yolo] creating ONNX session', modelUrl, executionProviders);

  // Create and return the session
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders,
    graphOptimizationLevel: 'all'
  });

  console.log('[yolo] session ready. inputs:', session.inputNames, 'outputs:', session.outputNames);
  return session;
}

// letterbox: resize + pad into target size (like your Python letterbox)

export function letterboxJS(image, targetW = WASM_DEFAULT_INPUT_SIZE, targetH = WASM_DEFAULT_INPUT_SIZE, color = [114,114,114]) {
  const srcW = image.videoWidth || image.width;
  const srcH = image.videoHeight || image.height;

  const r = Math.min(targetW / srcW, targetH / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);

  const dw = targetW - newW;
  const dh = targetH - newH;
  const padX = Math.round(dw / 2);
  const padY = Math.round(dh / 2);

  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.fillRect(0, 0, targetW, targetH);

  // draw resized image centered (handles video or image element)
  ctx.drawImage(image, 0, 0, srcW, srcH, padX, padY, newW, newH);

  return { canvas, scale: r, padX, padY, targetW, targetH };
}

// NMS helpers (greedy, same algorithm as your Python non_max_suppression)
function iou(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);
  const w = Math.max(0, x2 - x1);
  const h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
  const areaB = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
  const union = areaA + areaB - inter + 1e-16;
  return inter / union;
}

export function nonMaxSuppression(boxes, scores, iouThresh = 0.45) {
  if (!boxes || boxes.length === 0) return [];
  const idxs = scores.map((s,i) => i).sort((a,b) => scores[b] - scores[a]);
  const keep = [];
  while (idxs.length) {
    const i = idxs.shift();
    keep.push(i);
    const rest = [];
    for (const j of idxs) {
      if (iou(boxes[i], boxes[j]) <= iouThresh) rest.push(j);
    }
    idxs.splice(0, idxs.length, ...rest);
  }
  return keep;
}

// processOutputsJS: mirror your Python process_detections exactly (no sigmoid)
export function processOutputsJS(outputs, meta, conf_thresh = 0.25, iou_thresh = 0.45) {
  const detections = [];
  if (!outputs) return detections;

  // unify outputs: if outputs is an object with named tensors or an array, pick the
  // appropriate tensor. Commonly ORT returns an object with a tensor; handle that.
  let outTensor = null;
  if (Array.isArray(outputs)) outTensor = outputs[0];
  else if (outputs && outputs.data && outputs.dims) outTensor = outputs;
  else {
    const keys = Object.keys(outputs || {});
    if (keys.length > 0) outTensor = outputs[keys[0]];
  }

  if (!outTensor || !outTensor.data || !outTensor.dims) return detections;

  const data = outTensor.data;
  const dims = outTensor.dims; // [1, N, stride]
  if (!Array.isArray(dims) || dims.length < 3) return detections;

  const N = dims[1];
  const stride = dims[2];
  const numClasses = stride - 5;
  if (numClasses <= 0) return detections;

  const boxes = [];
  const scores = [];
  const classIds = [];

  // Exactly like Python: take cx,cy,w,h,obj_conf,cls_scores -> argmax -> cls_conf -> final = obj_conf * cls_conf
  for (let i = 0; i < N; i++) {
    const base = i * stride;
    const cx = data[base + 0];
    const cy = data[base + 1];
    const w = data[base + 2];
    const h = data[base + 3];
    const objConf = data[base + 4];

    // find top class and its score (no sigmoid applied; mirror Python)
    let bestClass = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const sc = data[base + 5 + c];
      if (sc > bestScore) { bestScore = sc; bestClass = c; }
    }
    const clsConf = bestScore;
    const finalScore = objConf * clsConf;

    if (finalScore < conf_thresh) continue;

    const x1 = cx - w/2;
    const y1 = cy - h/2;
    const x2 = cx + w/2;
    const y2 = cy + h/2;

    boxes.push([x1, y1, x2, y2]);
    scores.push(finalScore);
    classIds.push(bestClass);
  }

  if (!boxes.length) return detections;

  const keep = nonMaxSuppression(boxes, scores, iou_thresh);

  for (const idx of keep) {
    const [x1p, y1p, x2p, y2p] = boxes[idx];
    const sc = scores[idx];
    const cid = classIds[idx];

    const x1_unpad = (x1p - (meta.padX || meta.pad_x || 0)) / (meta.scale || 1);
    const y1_unpad = (y1p - (meta.padY || meta.pad_y || 0)) / (meta.scale || 1);
    const x2_unpad = (x2p - (meta.padX || meta.pad_x || 0)) / (meta.scale || 1);
    const y2_unpad = (y2p - (meta.padY || meta.pad_y || 0)) / (meta.scale || 1);

    const xmin = Math.max(0, Math.min(1, x1_unpad / (meta.orig_w || meta.img_size || 1)));
    const ymin = Math.max(0, Math.min(1, y1_unpad / (meta.orig_h || meta.img_size || 1)));
    const xmax = Math.max(0, Math.min(1, x2_unpad / (meta.orig_w || meta.img_size || 1)));
    const ymax = Math.max(0, Math.min(1, y2_unpad / (meta.orig_h || meta.img_size || 1)));

    detections.push({ label: parseInt(cid), score: Number(sc), xmin, ymin, xmax, ymax });
  }

  return detections;
}

// draw detections on canvas context (client-side overlay)
export function drawDetectionsOnCtx(ctx, detections, names = names_map) {
  if (!ctx || !ctx.canvas) return;
  const canvas = ctx.canvas;
  const W = canvas.width, H = canvas.height;
  ctx.textBaseline = 'top';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 2;

  for (const d of detections) {
    const x = d.xmin * W;
    const y = d.ymin * H;
    const w = (d.xmax - d.xmin) * W;
    const h = (d.ymax - d.ymin) * H;
    ctx.strokeStyle = '#00FF00';
    ctx.fillStyle = '#FF9900';
    ctx.strokeRect(x, y, w, h);

    const labelText = (names && names[d.label]) ? `${names[d.label]} ${ (d.score*100).toFixed(1)}%` : `${d.label} ${(d.score*100).toFixed(1)}%`;
    ctx.font = '14px Arial';
    const tm = ctx.measureText(labelText);
    ctx.fillRect(x, Math.max(0, y - 18), tm.width + 8, 18);
    ctx.fillStyle = '#000000';
    ctx.fillText(labelText, x + 4, Math.max(0, y - 16));
  }
}

// preprocessImageForSession: produce ort.Tensor [1,3,H,W] (NCHW), same as server
export async function preprocessImageForSession(imageElement, session, fallbackSize = WASM_DEFAULT_INPUT_SIZE) {
  // detect input size from session metadata similar to your previous logic
  let inputSize = fallbackSize;
  try {
    if (session && session.inputNames && session.inputMetadata) {
      const name = session.inputNames[0];
      const meta = session.inputMetadata[name];
      if (meta && meta.dimensions && meta.dimensions.length === 4) {
        const dims = meta.dimensions;
        const nums = dims.filter(d => typeof d === 'number');
        if (nums.length >= 2) {
          inputSize = Math.max(nums[nums.length-1], nums[nums.length-2]);
        }
      }
    }
  } catch (e) {
    // ignore
  }

  const { canvas, scale, padX, padY, targetW, targetH } = letterboxJS(imageElement, inputSize, inputSize);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  const area = canvas.width * canvas.height;
  const r = new Float32Array(area);
  const g = new Float32Array(area);
  const b = new Float32Array(area);
  let ptr = 0;
  for (let i = 0; i < data.length; i += 4) {
    r[ptr] = data[i] / 255.0;
    g[ptr] = data[i+1] / 255.0;
    b[ptr] = data[i+2] / 255.0;
    ptr++;
  }
  const transposed = new Float32Array(area * 3);
  transposed.set(r, 0);
  transposed.set(g, area);
  transposed.set(b, area * 2);

  const tensor = new ort.Tensor('float32', transposed, [1, 3, canvas.height, canvas.width]);

  const meta = {
    orig_w: imageElement.videoWidth || imageElement.width,
    orig_h: imageElement.videoHeight || imageElement.height,
    scale,
    padX,
    padY,
    img_size: Math.max(canvas.width, canvas.height)
  };

  return { tensor, meta, preprocessedCanvas: canvas };
}

// runInference wrapper
export async function runInference(session, imageElement, conf_thresh = 0.25, iou_thresh = 0.45) {
  const start = performance.now();
  const { tensor, meta, preprocessedCanvas } = await preprocessImageForSession(imageElement, session);
  const feeds = {};
  try {
    const inputName = session.inputNames && session.inputNames[0] ? session.inputNames[0] : Object.keys(session.inputMetadata || {})[0];
    if (inputName) feeds[inputName] = tensor;
    else feeds[session.inputNames[0]] = tensor;
  } catch (e) {
    feeds[session.inputNames[0] || 'images'] = tensor;
  }

  const outputs = await session.run(feeds);
  const detections = processOutputsJS(outputs, meta, conf_thresh, iou_thresh);
  const processingTime = performance.now() - start;

  return { detections, outputs, processingTime, preprocessedCanvas, meta };
}
