"""Failure-recovery environment on a frozen UETrack / ORTrack (batched over videos).

Every frame, the official step runs first (search 4x the target, Hanning window); its
score map gives the observation, and the policy then chooses how to use it:

  0 normal       the official box
  1 wide15       re-run with a 1.5x larger search region
  2 wide2        re-run with a 2x larger search region
  3 nowindow     same network output, no Hanning window
  4 halfwindow   same network output, window at half strength
  5 hold         keep the previous box
  6 extrapolate  previous box moved by the last frame-to-frame motion

(the actions of oracle_headroom.py). In training, each policy tracker has a paired
official tracker started from the same frame and template; the reward is
IoU(policy) - IoU(official) - WIDE_COST for wide searches, so the reward reflects only
the decisions. Evaluation tracks whole videos and reports both trackers per video.
"""
import json
import os

import cv2
import numpy as np
import torch

ACTIONS = ['normal', 'wide15', 'wide2', 'nowindow', 'halfwindow', 'hold', 'extrapolate']
WIDE = {1: 1.5, 2: 2.0}
WINDOW = {3: 0.0, 4: 0.5}
N_SCALARS = 18
OBS_DIM = N_SCALARS + len(ACTIONS)


def load_videos(root, json_name):
    with open(os.path.join(root, json_name)) as f:
        meta = json.load(f)
    return [{'name': k, 'frames': [os.path.join(root, p) for p in v['img_names']],
             'gt': np.nan_to_num(np.array(v['gt_rect'], dtype=np.float64)), 'attr': v.get('attr', [])}
            for k, v in meta.items()]


def iou(a, b):
    """a, b: (N, 4) x, y, w, h; zero where b has no valid box."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    out = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.)
    return np.where(b[:, 2] * b[:, 3] > 0, out, 0.)


def success_auc(boxes, gt):
    o = iou(np.asarray(boxes, dtype=np.float64), gt)
    return float(np.mean([(o > t).mean() for t in np.linspace(0, 1, 21)]))


class FREnv(object):
    def __init__(self, backend, videos, num_envs, episode_len=300, train=True, paired=True,
                 seed=0, wide_cost=0.002, init_jitter=0.0):
        self.b = backend
        self.videos = videos
        self.n = num_envs
        self.k = 2 * num_envs if paired else num_envs
        self.paired = paired
        self.train = train
        self.episode_len = episode_len
        self.wide_cost = wide_cost
        self.init_jitter = init_jitter
        self.rng = np.random.RandomState(seed)
        self.queue = list(range(len(videos)))          # eval: videos still to start
        k = self.k
        self.active = np.zeros(k, bool)
        self.video = np.zeros(k, int)
        self.t = np.zeros(k, int)
        self.end = np.zeros(k, int)
        self.state = np.zeros((k, 4))
        self.prev_state = np.zeros((k, 4))
        self.templates = [None] * k
        self.boxes = [None] * k
        self.actions = [None] * k
        self.ema = np.zeros(k)
        self.peaks = np.zeros((k, 3))
        self.since = np.zeros(k)
        self.prev_a = np.zeros(k, int)
        self.cache = None

    # ----------------------------------------------------------------- episodes
    def _image(self, v, t, memo):
        key = (v, t)
        if key not in memo:
            memo[key] = cv2.cvtColor(cv2.imread(self.videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
        return memo[key]

    def _start(self, i, memo):
        """Start policy slot i (and its shadow) on a new episode; False if none left."""
        if self.train:
            v = self.rng.randint(len(self.videos))
            gt = self.videos[v]['gt']
            ok = np.where((gt[:-1, 2] > 0) & (gt[:-1, 3] > 0))[0]
            if not len(ok):
                return self._start(i, memo)
            s = int(self.rng.choice(ok))
            e = min(len(gt) - 1, s + self.episode_len)
        else:
            if not self.queue:
                return False
            v, s = self.queue.pop(0), 0
            e = len(self.videos[v]['gt']) - 1
        box0 = self.videos[v]['gt'][s].copy()
        if self.init_jitter:
            # centre shift, sigma relative to the target size (sqrt(w*h))
            box0[:2] += self.rng.normal(0, self.init_jitter * np.sqrt(max(box0[2] * box0[3], 1.)), 2)
        tmpl = self.b.template(self._image(v, s, memo), box0)
        for j in ([i, i + self.n] if self.paired else [i]):
            self.active[j] = True
            self.video[j], self.t[j], self.end[j] = v, s + 1, e
            self.state[j] = self.prev_state[j] = box0
            self.templates[j] = tmpl
            self.boxes[j] = [box0.tolist()]
            self.actions[j] = []
            self.ema[j], self.peaks[j], self.since[j], self.prev_a[j] = 0., 0., 1., 0
        return True

    def reset(self):
        memo = {}
        self.active[:] = False
        for i in range(self.n):
            if not self._start(i, memo):
                break
        return self._observe(memo)

    # ----------------------------------------------------------------- observation
    def _observe(self, memo):
        ids = np.where(self.active)[0]
        if not len(ids):
            self.cache = None
            return np.zeros((0, OBS_DIM), np.float32)
        imgs = [self._image(self.video[j], self.t[j], memo) for j in ids]
        crops = [self.b.crop(im, self.state[j], self.b.p.search_factor) for im, j in zip(imgs, ids)]
        score, size, offset = self.b.forward(crops, [self.templates[j] for j in ids])
        box01, peak = self.b.decode(score, size, offset, np.ones(len(ids)))
        normal = np.array([self.b.box_back(box01[r], self.state[j], crops[r][1], imgs[r].shape[:2])
                           for r, j in enumerate(ids)])
        feats = self._features(ids, score, peak, normal, imgs)
        self.cache = {'ids': ids, 'imgs': imgs, 'crops': crops, 'score': score, 'size': size,
                      'offset': offset, 'normal': normal, 'peak': peak}
        pol = ids < self.n
        return feats[pol]

    def _features(self, ids, score, peak, normal, imgs):
        S = score.shape[-1]
        s = score.flatten(1)
        smax, arg = s.max(1)
        ys, xs = (arg // S).float(), (arg % S).float()
        yy, xx = torch.meshgrid(torch.arange(S, device=s.device), torch.arange(S, device=s.device), indexing='ij')
        far = ((yy.flatten()[None] - ys[:, None]).abs() > 2) | ((xx.flatten()[None] - xs[:, None]).abs() > 2)
        second = (s * far).max(1)[0] / smax.clamp_min(1e-6)
        p = s / s.sum(1, keepdim=True).clamp_min(1e-6)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(S * S)
        ctr = (((ys - (S - 1) / 2) ** 2 + (xs - (S - 1) / 2) ** 2).sqrt() / (S / 2))
        smax, second, ent, ctr = [x.cpu().numpy() for x in (smax, second, ent, ctr)]
        pk = peak
        st = self.state[ids]
        out = np.zeros((len(ids), OBS_DIM), np.float32)
        for r, j in enumerate(ids):
            w, h = max(st[r, 2], 1.), max(st[r, 3], 1.)
            H, W = imgs[r].shape[:2]
            ema = self.ema[j] if self.ema[j] > 0 else pk[r]
            dc = np.hypot(normal[r, 0] + normal[r, 2] / 2 - st[r, 0] - w / 2,
                          normal[r, 1] + normal[r, 3] / 2 - st[r, 1] - h / 2) / np.sqrt(w * h)
            cx, cy = st[r, 0] + w / 2, st[r, 1] + h / 2
            border = min(cx, cy, W - cx, H - cy) / np.sqrt(w * h)
            out[r, :N_SCALARS] = [
                smax[r], pk[r], second[r], ent[r], ctr[r],
                np.log(max(normal[r, 2], 1.) / w), np.log(max(normal[r, 3], 1.) / h),
                min(dc, 5.), pk[r] / max(ema, 1e-6), ema,
                self.peaks[j, 0], self.peaks[j, 1], self.peaks[j, 2],
                min(self.since[j] / 50., 1.), np.sqrt(w * h / (H * W)), min(border, 5.),
                np.log(w / h), float(self.prev_a[j] in (5, 6))]
            out[r, N_SCALARS + self.prev_a[j]] = 1.
        return out

    # ----------------------------------------------------------------- step
    def step(self, actions):
        """actions: (n_active_policy,) ints. Returns obs, reward, done, finished (eval)."""
        c = self.cache
        ids, normal = c['ids'], c['normal']
        pol = ids < self.n
        act = np.zeros(len(ids), int)
        act[pol] = actions
        new = normal.copy()
        # window variants: same network output, decoded again
        for a, wv in WINDOW.items():
            rows = np.where(act == a)[0]
            if len(rows):
                b01, _ = self.b.decode(c['score'][rows], c['size'][rows], c['offset'][rows], np.full(len(rows), wv))
                for q, r in enumerate(rows):
                    new[r] = self.b.box_back(b01[q], self.state[ids[r]], c['crops'][r][1], c['imgs'][r].shape[:2])
        # wide searches: one extra batched forward
        rows = np.where(np.isin(act, list(WIDE)))[0]
        if len(rows):
            crops = [self.b.crop(c['imgs'][r], self.state[ids[r]], self.b.p.search_factor * WIDE[act[r]]) for r in rows]
            sc, sz, of = self.b.forward(crops, [self.templates[ids[r]] for r in rows])
            b01, _ = self.b.decode(sc, sz, of, np.ones(len(rows)))
            for q, r in enumerate(rows):
                new[r] = self.b.box_back(b01[q], self.state[ids[r]], crops[q][1], c['imgs'][r].shape[:2])
        for r in np.where(act == 5)[0]:
            new[r] = self.state[ids[r]]
        for r in np.where(act == 6)[0]:
            j = ids[r]
            d = self.state[j, :2] - self.prev_state[j, :2]
            H, W = c['imgs'][r].shape[:2]
            new[r] = self.b.clip_box([self.state[j, 0] + d[0], self.state[j, 1] + d[1],
                                      self.state[j, 2], self.state[j, 3]], H, W, margin=10)

        gt = np.stack([self.videos[self.video[j]]['gt'][self.t[j]] for j in ids])
        ov = iou(new, gt)
        reward = np.zeros(self.n)
        if self.paired:
            pos = {j: r for r, j in enumerate(ids)}
            for r, j in enumerate(ids):
                if j < self.n:
                    reward[j] = ov[r] - ov[pos[j + self.n]] - (self.wide_cost if act[r] in WIDE else 0.)
        # advance
        for r, j in enumerate(ids):
            self.prev_state[j], self.state[j] = self.state[j].copy(), new[r]
            self.boxes[j].append(new[r].tolist())
            self.actions[j].append(int(act[r]))
            pk = c['peak'][r]
            self.ema[j] = pk if self.ema[j] == 0 else 0.9 * self.ema[j] + 0.1 * pk
            self.peaks[j] = [pk, self.peaks[j, 0], self.peaks[j, 1]]
            self.since[j] = 0 if act[r] != 0 else self.since[j] + 1
            self.prev_a[j] = act[r]
            self.t[j] += 1
        done = np.zeros(self.n, bool)
        finished = []
        memo = {}
        for i in range(self.n):
            if not self.active[i] or self.t[i] <= self.end[i]:
                continue
            done[i] = True
            if not self.train:
                v = self.videos[self.video[i]]
                rec = {'name': v['name'], 'boxes': self.boxes[i], 'actions': self.actions[i],
                       'auc': success_auc(self.boxes[i], v['gt'])}
                if self.paired:
                    rec['official_boxes'] = self.boxes[i + self.n]
                    rec['official_auc'] = success_auc(self.boxes[i + self.n], v['gt'])
                finished.append(rec)
            for j in ([i, i + self.n] if self.paired else [i]):
                self.active[j] = False
            self._start(i, memo)
        obs = self._observe(memo)
        return obs, reward, done, finished

    def policy_slots(self):
        """Policy slot ids, in the order of the rows of the last observation."""
        return self.cache['ids'][self.cache['ids'] < self.n] if self.cache is not None else np.zeros(0, int)
