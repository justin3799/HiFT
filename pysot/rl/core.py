"""HiFT decoding and post-processing, batched over several search crops.

The maths matches HiFTTracker.track exactly when the action equals
baseline_action(); the RL policy replaces the four fixed constants
(window influence, scale penalty k, size lr, search scale) per frame.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.functional as F

from pysot.core.config import cfg

ACTION_NAMES = ['window_influence', 'penalty_k', 'lr', 'search_scale']
# summary statistics in make_obs, plus the previous action
NUM_SCALARS = 15 + len(ACTION_NAMES)


def action_bounds():
    return (np.array(cfg.RL.ACTION_LOW, dtype=np.float64),
            np.array(cfg.RL.ACTION_HIGH, dtype=np.float64))


def baseline_action():
    """The hand-tuned constants HiFTTracker uses, as an action."""
    return np.array([cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.PENALTY_K,
                     cfg.TRACK.LR, 1.0], dtype=np.float64)


def to_unit(action):
    low, high = action_bounds()
    return (action - low) / (high - low)


def from_unit(a01):
    low, high = action_bounds()
    return low + (high - low) * a01


def obs_dim():
    maps = 2 * cfg.TRAIN.OUTPUT_SIZE ** 2 if cfg.RL.OBS_MAPS else 0
    return maps + NUM_SCALARS


def hanning_window():
    h = np.hanning(cfg.TRAIN.OUTPUT_SIZE)
    return np.outer(h, h).flatten()


def search_region(size, scaleaa, img_hw, scale_factor):
    """Search crop side in image pixels and image->crop scale, per sample.

    size (B,2), scaleaa (B,), img_hw (B,2), scale_factor (B,)
    """
    ctx = cfg.TRACK.CONTEXT_AMOUNT * size.sum(1)
    s_z = np.sqrt((size[:, 0] + ctx) * (size[:, 1] + ctx))
    if cfg.RL.TRACKER == 'HiFT':
        # HiFT keeps the first frame's crop size once the target fills half the image
        big = size[:, 0] * size[:, 1] > 0.5 * img_hw[:, 0] * img_hw[:, 1]
        s_z = np.where(big, scaleaa, s_z)
    scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE) * scale_factor
    return np.round(s_x), scale_z / scale_factor


def decode(outputs):
    """Network outputs for B crops -> candidate boxes and scores.

    Returns numpy arrays:
        pred     (B,4,N) [cx, cy, w, h] in crop pixels, relative to the crop centre
        score    (B,N)   fused score the tracker ranks candidates by
        cls_prob (B,N)   inside-box probability
        ctr_prob (B,N)   centre-closeness probability
    """
    if cfg.RL.TRACKER == 'SiamAPN++':
        return _decode_apn(outputs)
    return _decode_hift(outputs)


def _decode_apn(outputs):
    """As ADSiamAPNTracker.track: anchors from the anchor proposal network,
    refined by the regression head; score from three classifiers."""
    anchors, loc = outputs['anchors'], outputs['loc']
    b, _, size, _ = loc.shape
    stride = cfg.ANCHOR.STRIDE
    grid = stride * np.arange(size) - stride * (size - 1) / 2
    shap = (anchors * (cfg.TRAIN.SEARCH_SIZE // 4)).cpu().numpy()
    aw = shap[:, 0] + shap[:, 1]
    ah = shap[:, 2] + shap[:, 3]
    ax = grid[None, None, :] - shap[:, 0] + aw / 2
    ay = grid[None, :, None] - shap[:, 2] + ah / 2
    aw, ah, ax, ay = [a.reshape(b, -1) for a in (aw, ah, ax, ay)]
    d = loc.reshape(b, 4, -1).cpu().numpy()
    # the reference writes each row back into a float32 array
    pred = np.stack([(d[:, 0] * aw + ax).astype(np.float32),
                     (d[:, 1] * ah + ay).astype(np.float32),
                     (np.exp(d[:, 2]) * aw).astype(np.float32),
                     (np.exp(d[:, 3]) * ah).astype(np.float32)], 1).astype(np.float64)

    p1 = F.softmax(outputs['cls1'], dim=1)[:, 1].reshape(b, -1)
    p2 = F.softmax(outputs['cls2'], dim=1)[:, 1].reshape(b, -1)
    ctr_logit = outputs['cls3'].reshape(b, -1)
    score = (p1 * cfg.TRACK.w1 + p2 * cfg.TRACK.w2 + ctr_logit * cfg.TRACK.w3) / 3
    return (pred,
            score.cpu().numpy().astype(np.float64),
            ((p1 + p2) / 2).cpu().numpy(),
            torch.sigmoid(ctr_logit).cpu().numpy())


def _decode_hift(outputs):
    loc = outputs['loc']
    b, _, size, _ = loc.shape
    half = cfg.TRAIN.SEARCH_SIZE // 2
    d = torch.where(loc <= -1, torch.full_like(loc, -0.99), loc)
    d = torch.where(d >= 1, torch.full_like(d, 0.99), d)
    shap = torch.atanh(d) * 143
    grid = (cfg.ANCHOR.STRIDE * torch.arange(size, device=loc.device, dtype=loc.dtype)
            + 63 - half)
    w = shap[:, 0] + shap[:, 1]
    h = shap[:, 2] + shap[:, 3]
    cx = grid.view(1, 1, size) - shap[:, 0] + w / 2
    cy = grid.view(1, size, 1) - shap[:, 2] + h / 2
    pred = torch.stack([cx, cy, w, h], 1).view(b, 4, -1)

    cls_prob = F.softmax(outputs['cls1'], dim=1)[:, 1].reshape(b, -1)
    ctr_logit = outputs['cls2'].reshape(b, -1)
    score = (cls_prob * cfg.TRACK.w2 + ctr_logit * cfg.TRACK.w3) / 2
    return (pred.cpu().numpy().astype(np.float64),
            score.cpu().numpy().astype(np.float64),
            cls_prob.cpu().numpy(),
            torch.sigmoid(ctr_logit).cpu().numpy())


def _change(r):
    return np.maximum(r, 1. / (r + 1e-5))


def _sz(w, h):
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))


def postprocess(pred, score, window, size_crop, action):
    """Pick the best candidate with HiFT's penalties, parameterised by action.

    pred (B,4,N), score (B,N), size_crop (B,2) previous target size in crop
    pixels, action (B,4) in real units. Returns box (B,4) in crop pixels,
    size learning rate (B,), and the raw score of the chosen candidate (B,).
    """
    w, h = pred[:, 2], pred[:, 3]
    pw, ph = size_crop[:, 0:1], size_crop[:, 1:2]
    s_c = _change(_sz(w, h) / _sz(pw, ph))
    r_c = _change((pw / (ph + 1e-5)) / (w / (h + 1e-5)))
    penalty = np.exp(-(r_c * s_c - 1) * action[:, 1:2])
    pscore = penalty * score
    pscore = pscore * (1 - action[:, 0:1]) + window[None] * action[:, 0:1]
    best = np.argmax(pscore, 1)
    idx = np.arange(len(best))
    box = pred[idx, :, best]
    lr = penalty[idx, best] * score[idx, best] * action[:, 2]
    return box, lr, score[idx, best]


def update_state(center, size, box, scale, lr, img_hw):
    """Move the target state to the chosen box, as HiFTTracker does."""
    bbox = box / scale[:, None]
    cx = bbox[:, 0] + center[:, 0]
    cy = bbox[:, 1] + center[:, 1]
    width = size[:, 0] * (1 - lr) + bbox[:, 2] * lr
    height = size[:, 1] * (1 - lr) + bbox[:, 3] * lr
    cx = np.maximum(0, np.minimum(cx, img_hw[:, 1]))
    cy = np.maximum(0, np.minimum(cy, img_hw[:, 0]))
    width = np.maximum(10, np.minimum(width, img_hw[:, 1]))
    height = np.maximum(10, np.minimum(height, img_hw[:, 0]))
    return np.stack([cx, cy], 1), np.stack([width, height], 1)


def make_obs(pred, score, cls_prob, ctr_prob, size_crop, scale_factor,
             size, img_hw, prev_action01, score_ema, prev_max):
    """Policy input: both score maps plus summary statistics of this frame."""
    b, n = score.shape
    side = cfg.TRAIN.OUTPUT_SIZE
    c = (side - 1) / 2
    idx = np.arange(b)
    best = np.argmax(score, 1)
    smax = score[idx, best]
    row, col = best // side, best % side
    # strongest candidate away from the peak (distractor check)
    rr, cc = np.arange(n) // side, np.arange(n) % side
    near = (np.abs(rr[None] - row[:, None]) <= 1) & (np.abs(cc[None] - col[:, None]) <= 1)
    second = np.where(near, -np.inf, score).max(1)
    psr = (smax - score.mean(1)) / (score.std(1) + 1e-6)
    bw = np.maximum(pred[idx, 2, best], 1e-3)
    bh = np.maximum(pred[idx, 3, best], 1e-3)
    area = size[:, 0] * size[:, 1] / (img_hw[:, 0] * img_hw[:, 1])
    scalars = np.stack([
        smax,
        cls_prob.max(1),
        ctr_prob.max(1),
        second - smax,
        psr / 10,
        (col - c) / c,
        (row - c) / c,
        np.clip(np.log(bw / size_crop[:, 0]), -3, 3),
        np.clip(np.log(bh / size_crop[:, 1]), -3, 3),
        scale_factor,
        np.log(area) / 10,
        np.log(size[:, 0] / size[:, 1]),
        smax - score_ema,
        score_ema,
        prev_max,
    ], 1)
    parts = [cls_prob, ctr_prob] if cfg.RL.OBS_MAPS else []
    obs = np.concatenate(parts + [scalars, prev_action01], 1)
    return np.nan_to_num(obs.astype(np.float32), nan=0., posinf=5., neginf=-5.)
