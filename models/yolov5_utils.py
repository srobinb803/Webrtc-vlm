import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, int, int]:
    """Resize and pad image to target size while maintaining aspect ratio.
    Returns (padded_img_bgr, scale, pad_x, pad_y)
    """
    shape = im.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Apply Non-Maximum Suppression to filter overlapping boxes."""
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def process_detections(outputs: List[np.ndarray], meta: Dict[str, Any], conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Process YOLO outputs and return formatted detections with normalized coords 0..1."""
    if not outputs or outputs[0] is None:
        return []
    out = outputs[0]
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.shape[1] < 6:
        return []
    xywh = out[:, 0:4].astype(float)
    obj_conf = out[:, 4].astype(float)
    cls_scores = out[:, 5:].astype(float)
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
    scores = obj_conf * cls_conf
    keep_mask = scores >= conf_thresh
    if not np.any(keep_mask):
        return []
    xywh = xywh[keep_mask]; scores = scores[keep_mask]; cls_ids = cls_ids[keep_mask]
    cx = xywh[:, 0]; cy = xywh[:, 1]; w = xywh[:, 2]; h = xywh[:, 3]
    x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
    boxes_padded = np.stack([x1, y1, x2, y2], axis=1)
    keep_inds = non_max_suppression(boxes_padded, scores, iou_thresh)
    boxes_nms = boxes_padded[keep_inds]; scores_nms = scores[keep_inds]; cls_nms = cls_ids[keep_inds]
    detections = []
    for bx, sc, cid in zip(boxes_nms, scores_nms, cls_nms):
        x1_p, y1_p, x2_p, y2_p = bx.tolist()
        x1_unpad = (x1_p - meta['pad_x']) / meta['scale']
        y1_unpad = (y1_p - meta['pad_y']) / meta['scale']
        x2_unpad = (x2_p - meta['pad_x']) / meta['scale']
        y2_unpad = (y2_p - meta['pad_y']) / meta['scale']
        xmin = max(0.0, min(1.0, x1_unpad / meta['orig_w']))
        ymin = max(0.0, min(1.0, y1_unpad / meta['orig_h']))
        xmax = max(0.0, min(1.0, x2_unpad / meta['orig_w']))
        ymax = max(0.0, min(1.0, y2_unpad / meta['orig_h']))
        detections.append({
            "label": int(cid),
            "score": float(sc),
            "xmin": float(xmin),
            "ymin": float(ymin),
            "xmax": float(xmax),
            "ymax": float(ymax)
        })
    return detections

def draw_detections(img: np.ndarray, detections: List[Dict[str, Any]], names_map: Dict[int, str]) -> np.ndarray:
    """Draw detection bounding boxes and labels on a BGR image."""
    overlay = img.copy()
    h, w = img.shape[:2]
    for d in detections:
        xmin = int(d['xmin'] * w); ymin = int(d['ymin'] * h)
        xmax = int(d['xmax'] * w); ymax = int(d['ymax'] * h)
        label_id = int(d['label']); score = d.get('score', 0.0)
        label = names_map.get(label_id, str(label_id))
        color = (0, 255, 0)  # green (B, G, R)
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
        caption = f"{label} {score:.2f}"
        tsize = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(overlay, (xmin, max(0, ymin-tsize[1]-6)), (xmin+tsize[0]+6, ymin), color, -1)
        cv2.putText(overlay, caption, (xmin+3, max(0, ymin-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
names_map = {i: n for i, n in enumerate(COCO_NAMES)}
