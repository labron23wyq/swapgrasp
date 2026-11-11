"""Canonical xywh-theta geometry; no label-dependent inference filtering."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def angle_error(a, b):
    return abs((float(a) - float(b) + math.pi / 2) % math.pi - math.pi / 2)


def corners(g):
    x, y, w, h, theta = map(float, g)
    u = np.array([math.cos(theta), math.sin(theta)]) * w / 2
    v = np.array([-math.sin(theta), math.cos(theta)]) * h / 2
    c = np.array([x, y])
    return np.asarray([c-u-v, c+u-v, c+u+v, c-u+v], dtype=np.float32)


def rotated_iou(a, b):
    if min(a[2], a[3], b[2], b[3]) <= 0:
        return 0.0
    area, _ = cv2.intersectConvexConvex(corners(a), corners(b))
    return float(np.clip(area / max(a[2]*a[3] + b[2]*b[3] - area, 1e-9), 0, 1))


def grasp_success(preds, grasps, k=1, iou=0.25, angle_degrees=30):
    # Both conditions must hold for the SAME GT rectangle.
    return any(angle_error(p['grasp'][4], g[4]) < math.radians(angle_degrees)
               and rotated_iou(p['grasp'], g) > iou
               for p in preds[:k] for g in grasps)


def transform_grasps(grasps, transform, inverse=False):
    result = np.array(grasps, dtype=np.float32).reshape(-1, 5).copy()
    scale, left, top = transform['scale'], transform['left'], transform['top']
    if inverse:
        result[:, :2] = (result[:, :2] - [left, top]) / scale
        result[:, 2:4] /= scale
    else:
        result[:, :2] = result[:, :2] * scale + [left, top]
        result[:, 2:4] *= scale
    return result


def decode(output, valid, topk=5, candidates=50):
    """output [5,H,W], valid [H,W]; accesses no GT or target mask."""
    output = output.detach().float()
    height, width = output.shape[-2:]
    quality = output[0].sigmoid().masked_fill(~valid.bool(), -1)
    local = F.max_pool2d(quality[None, None], 3, 1, 1)[0, 0]
    peaks = quality.masked_fill(quality != local, -1)
    scores, indexes = peaks.flatten().topk(min(candidates, peaks.numel()))
    values = output.cpu().numpy()
    accepted = []
    for score, index in zip(scores.cpu().tolist(), indexes.cpu().tolist()):
        if score < 0:
            continue
        y, x = divmod(index, width)
        c, s, lw, lh = values[1:, y, x]
        if not np.isfinite([c, s, lw, lh]).all() or np.hypot(c, s) < 1e-6:
            continue
        w = width * np.exp(np.clip(lw, math.log(1 / width), math.log(2)))
        h = height * np.exp(np.clip(lh, math.log(1 / height), math.log(2)))
        g = [float(x), float(y), float(w), float(h), float(0.5*np.arctan2(s, c))]
        if any(angle_error(g[4], p['grasp'][4]) < math.radians(15)
               and rotated_iou(g, p['grasp']) > 0.25 for p in accepted):
            continue
        accepted.append({'grasp': g, 'score': float(score)})
        if len(accepted) == topk:
            break
    return accepted

